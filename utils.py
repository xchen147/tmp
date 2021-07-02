"""Code to perform composition optimization

Author: Xi Chen
"""
import numpy as np
from joblib import load
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from copy import deepcopy
from tqdm.notebook import tqdm
import shap
from pymatgen.core import periodic_table

"""Setting up. """

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
Ele = periodic_table.Element
ele_list = np.array([str(Ele.from_Z(i)) for i in range(1,104)])
def dict_to_vec(dict_):
    vec = np.zeros(103)
    for key,value in dict_.items():
        vec[np.where(ele_list==key)[0]] = value
    return vec

ks = keras.models.load_model('./0V_models/li_upper_bound_25_lower_bound_0.2_kstar')
chem0v = keras.models.load_model('./0V_models/li_upper_bound_-0.1_lower_bound_-1.5_chem0')
kc = load('./4V_models/li_kcrit_model_4V')
chem4v = load('./4V_models/li_decomp_model_4V')

target = 'kstar'

if target == 'kstar':
    ip_data = pd.read_csv('./kstar_Li.csv')
else:
    ip_data = pd.read_csv('./chem_at0GPa_Li.csv')
comp = ip_data['electrolyte'].values
ele_to_num_table = {}
for i in range(103):
    ele_to_num_table.update({str(Ele.from_Z(i+1)):i})
def one_hot_comp(comp):
    ele_to_num_table = {}
    for i in range(103):
        ele_to_num_table.update({str(Ele.from_Z(i+1)):i})
    encoded = np.zeros((len(comp),103))
    for i in tqdm(range(len(comp))):
        species = comp[i].split()
        for j in range(len(species)):
            try:
                number = int(species[j][-3:])
                encoded[i][ele_to_num_table[species[j][:-3]]] = number
            except:
                try:
                    number = int(species[j][-2:])
                    encoded[i][ele_to_num_table[species[j][:-2]]] = number
                except:
                    number = int(species[j][-1])
                    encoded[i][ele_to_num_table[species[j][:-1]]] = number
    return encoded
numbered_comp = one_hot_comp(comp)
norm_comp = numbered_comp/np.sum(numbered_comp,axis=1).reshape(-1,1)
comp_selected = []
hull_selected = []
for i in range(len(ip_data.values[:,2:].T)):
    hull = (ip_data.values[:,2:].T)[i]
    comp_selected.append(np.hstack((norm_comp,i*np.ones((len(norm_comp),1)))))
    hull_selected.append(hull)
    
comp_selected = np.concatenate(comp_selected)
hull_selected = np.concatenate(hull_selected)

if target == 'kstar':
    comp_selected = comp_selected[hull_selected<25]
    hull_selected = hull_selected[hull_selected<25]
    comp_selected = comp_selected[hull_selected>0.2]
    hull_selected = hull_selected[hull_selected>0.2]
else:
    comp_selected = comp_selected[hull_selected<-0.1]
    hull_selected = hull_selected[hull_selected<-0.1]
    comp_selected = comp_selected[hull_selected>-1.5]
    hull_selected = hull_selected[hull_selected>-1.5]
X_train = comp_selected
background = X_train[np.random.choice(X_train.shape[0], 500, replace=False)]


def one_hot_comp_single(comp):
    '''Converting composition to vectors by one-hot encoding'''
    ele_to_num_table = {}
    for i in range(103):
        ele_to_num_table.update({str(Ele.from_Z(i+1)):i})
    
    encoded = np.zeros(103)
    species = comp.split()
    for j in range(len(species)):
        try:
            number = int(species[j][-3:])
            encoded[ele_to_num_table[species[j][:-3]]] = number
        except ValueError:
            try:
                number = int(species[j][-2:])
                encoded[ele_to_num_table[species[j][:-2]]] = number
            except ValueError:
                number = int(species[j][-1])
                encoded[ele_to_num_table[species[j][:-1]]] = number
    return encoded
def one_hot_comp(comp):
    ele_to_num_table = {}
    for i in range(103):
        ele_to_num_table.update({str(Ele.from_Z(i+1)):i})
    encoded = np.zeros((len(comp),103))
    element = [[] for i in range(len(comp))]
    for i in range(len(comp)):
        species = comp[i].split()
        for j in range(len(species)):
            try:
                number = float(species[j][-6:])
                encoded[i][ele_to_num_table[species[j][:-6]]] = number
                element[i].append(species[j][:-6])
            except:
                try:
                    number = float(species[j][-5:])
                    encoded[i][ele_to_num_table[species[j][:-5]]] = number
                    element[i].append(species[j][:-5])
                except:
                    try:
                        number = float(species[j][-4:])
                        encoded[i][ele_to_num_table[species[j][:-4]]] = number
                        element[i].append(species[j][:-4])
                    except:
                        try:
                            number = float(species[j][-3:])
                            encoded[i][ele_to_num_table[species[j][:-3]]] = number
                            element[i].append(species[j][:-3])
                        except:
                            try:
                                number = float(species[j][-2:])
                                encoded[i][ele_to_num_table[species[j][:-2]]] = number
                                element[i].append(species[j][:-2])
                            except:
                                number = float(species[j][-1])
                                encoded[i][ele_to_num_table[species[j][:-1]]] = number
                                element[i].append(species[j][:-1])
    return encoded[0], element[0]

def li_normalize(vec):
    return vec/np.sum(vec)

def ec_vector(id_):
    ## Produces the result needed based on the input id
    product = list(ec_product_query(id_).keys()) # id_'s
    ratio = list(ec_product_query(id_).values()) # the ratio of each decomp. phase
    vectors = []
    for i in range(len(product)):
#         print(product[i])
        comp = comp_query(product[i])
#         print(comp)
        encoded = one_hot_comp_single(comp)
        encoded = no_li_normalize(encoded)
        vectors.append(encoded)
    vectors = np.sum(np.array(vectors),axis = 0)
    return vectors

def ec_vector_with_li(id_):
    """ Produces the result needed based on the input id. """
    product = list(ec_product_query(id_).keys()) # id_'s
    ratio = list(ec_product_query(id_).values()) # the ratio of each decomp. phase
    vectors = []
    for i in range(len(product)):
#         print(product[i])
        comp = comp_query(product[i])
#         print(comp)
        encoded = one_hot_comp_single(comp)
        encoded = encoded/np.sum(encoded)
        vectors.append(encoded)
    vectors = np.sum(np.array(vectors),axis = 0)
    return vectors

def ec_vector_handinput(comp_amount_dict):
    return li_normalize(np.sum([amount*one_hot_comp_single(comp) for comp,
                   amount in comp_amount_dict.items()],axis=0))

def optimize_full_constraint(input_dict,constraint_level = 0.05,target = 'kstar',low_thresh = 0.05,
                            avail_ele = ['Li','Si','P','S','Cl']):
    """Main worker function. """
    vlow = deepcopy(input_dict)
    for key, val in vlow.items():
        vlow[key] = (1-constraint_level)*val
    vhigh = deepcopy(input_dict)
    for key, val in vhigh.items():
        vhigh[key] = np.min((1,(1+constraint_level)*val))
    cur_dict = deepcopy(input_dict)
    treated_elem = []
    all_dict = []
    if target == 'kstar':
        model_target = ks
    else:
        model_target = chem0v
    for i in range(20):
        init_value = li_normalize(dict_to_vec(input_dict))
        
        explainer = shap.DeepExplainer(model_target, background)
        
        x = np.argmax(model_target.predict(np.array([np.concatenate((li_normalize(dict_to_vec(cur_dict)),
                                                                     np.array([x]))) for x in range(10)])))
        
        shap_values = explainer.shap_values(np.concatenate((dict_to_vec(cur_dict),
                                                            np.array([x]))).reshape(1,104))[0][0][:-1]
        count = 0
        elem = ele_list[np.argsort(shap_values)[-1-count]]
        
        if len(avail_ele) == 2:
            while elem in treated_elem[-1:] or elem not in avail_ele:
                count += 1
                elem = ele_list[np.argsort(shap_values)[-1-count]]
        else:
            while elem in treated_elem[-2:] or elem not in avail_ele:
                count += 1
                elem = ele_list[np.argsort(shap_values)[-1-count]]
        treated_elem.append(elem)
        result = []
        result_vec = []
        for j in range(100):
            test = deepcopy(cur_dict)
            test[elem] = vlow[elem]+(vhigh[elem]-vlow[elem])*j/100
            if target == 'kstar':
                result.append(np.max(model_target.predict(np.array([np.concatenate((li_normalize(dict_to_vec(test)),
                                                                         np.array([x]))) for x in range(10)]))))
            else:
                result.append(np.min(model_target.predict(np.array([np.concatenate((li_normalize(dict_to_vec(test)),
                                                                         np.array([x]))) for x in range(10)]))))
            result_vec.append(li_normalize(dict_to_vec(test)).reshape(103))
        update_vec = result_vec[np.argmin(result)]
        cur_dict = dict(zip(ele_list[update_vec!=0],update_vec[update_vec!=0]))
        all_dict.append(cur_dict)
    to_del = []
    for elem, value in cur_dict.items():
        if value == 0:
            to_del.append(elem)
    for elem in to_del:
        del(cur_dict[elem])
    return all_dict


def get_optimization(names, constraint_level=0.5):
    """Performing a series of optimization"""
    optimized_dicts = []
    optimized_strings = []
    for i in tqdm(range(len(names))):
        vec, element = one_hot_comp(np.array([names[i]]))
        vec = vec/np.sum(vec)
        input_dict = dict(zip(ele_list[vec!=0],vec[vec!=0]))
        optimized_dict = [optimize_full_constraint(input_dict,constraint_level=constraint_level,avail_ele=element)]
        kcrit = np.array([np.max(ks.predict(np.array([np.concatenate((li_normalize(dict_to_vec(optimized_dict[0][i])),
                                                                         np.array([x]))) for x in range(10)]))) for i in range(20)])
        target = optimized_dict[0][np.argmin(kcrit)]
        optimized_dicts.append(target)
        string = ''
        for key in target.keys():
            string += key+str(target[key])[:6]+' '
        optimized_strings.append(string[:-1])
    return optimized_dicts, optimized_strings