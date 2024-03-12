import pandas as pd
df_layout = pd.read_csv('VOC2007/ImageSets/Layout/train.txt', header=None, dtype=str)
_cat = pd.read_csv('VOC2007/ImageSets/Main/cat_train.txt', header=None, dtype=str, delim_whitespace=True)
_cat.columns =['img','label']
df_cat = _cat[_cat['label'] == '1']
_dog = pd.read_csv('VOC2007/ImageSets/Main/dog_train.txt', header=None, dtype=str, delim_whitespace=True)
_dog.columns =['img','label']
df_dog = _dog[_dog['label'] == '1']
_cow = pd.read_csv('VOC2007/ImageSets/Main/cow_train.txt', header=None, dtype=str, delim_whitespace=True)
_cow.columns =['img','label']
df_cow = _cow[_cow['label'] == '1']
_bird = pd.read_csv('VOC2007/ImageSets/Main/bird_train.txt', header=None, dtype=str, delim_whitespace=True)
_bird.columns =['img','label']
df_bird = _bird[_bird['label'] == '1']
training_set = result = pd.concat([df_cat, df_dog, df_cow, df_bird])
#print(training_set)
