# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 19:54:38 2020

@author: shara
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
delivery = pd.read_csv("F:\Warun\DS Assignments\DS Assignments\SLR\delivery_time.csv")
deli = delivery
del(delivery)
print(deli.head)
plt.plot(deli.SortingTime, deli.DeliveryTime, "bo")
deli.columns = deli.columns.str.replace(' ','')
deli1 = deli.drop(deli.index[[20]], axis = 0)
print(deli1)
plt.plot(deli1.SortingTime, deli1.DeliveryTime, "bo")
plt.hist(deli1.DeliveryTime)
plt.boxplot(deli1.DeliveryTime)
import statsmodels.formula.api as smf
model = smf.ols(" DeliveryTime ~ SortingTime", data = deli).fit()
model.summary()
import statsmodels.api as sm
sm.graphics.plot_partregress_grid(model)
sm.graphics.influence_plot(model)
deli2 = deli1.drop(deli1.index[[16]], axis=0)
model2 = smf.ols(" DeliveryTime ~ SortingTime", data = deli2).fit()
sm.graphics.plot_partregress_grid(model2)
sm.graphics.influence_plot(model2)
model1 = smf.ols(" DeliveryTime ~ SortingTime", data = deli1).fit()
sm.graphics.plot_partregress_grid(model2)
sm.graphics.influence_plot(model2)
deli3 = deli1.drop(deli1.index[[16,12]], axis=0)
model3 = smf.ols(" DeliveryTime ~ SortingTime", data = deli3).fit()
sm.graphics.plot_partregress_grid(model2)
sm.graphics.influence_plot(model2)
deli4 = deli1.drop(deli1.index[[4,16,12]], axis=0)
model4 = smf.ols(" DeliveryTime ~ SortingTime", data = deli4).fit()
sm.graphics.plot_partregress_grid(model4)
sm.graphics.influence_plot(model4)
pred = model.predict(deli.iloc[ : , 1])
pred1 = model.predict(deli1.iloc[ : , 1])
pred2 = model.predict(deli2.iloc[ : , 1])
pred3 = model.predict(deli3.iloc[ : , 1])
pred4 = model.predict(deli3.iloc[ : , 1])
model.summary()
model1.summary()
model2.summary()
model3.summary()
model4.summary()
RMSE_model = np.sqrt(np.mean((deli.DeliveryTime - pred)**2))
RMSE_model1 = np.sqrt(np.mean((deli1.DeliveryTime - pred1)**2))
RMSE_model2 = np.sqrt(np.mean((deli2.DeliveryTime - pred2)**2))
RMSE_model3 = np.sqrt(np.mean((deli3.DeliveryTime - pred3)**2))
RMSE_model4 = np.sqrt(np.mean((deli4.DeliveryTime - pred4)**2))
print(RMSE_model)
print(RMSE_model1)
print(RMSE_model2)
print(RMSE_model3)
print(RMSE_model4)
plt.plot(model4.resid_pearson,"o")
