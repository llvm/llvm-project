// RUN: %clang_cc1 -E %s | grep 'ei_1 = (17 +1);'
// RUN: %clang_cc1 -E %s | grep 'ei_2 = (M1)(17);'

#define M1(a) (a+1) 
#define M2(b) b 

int ei_1 = M2(M1)(17); /* becomes int ei_1 = (17+1); */ 
int ei_2 = (M2(M1))(17); /* becomes int ei_2 = (M1)(17); */ 

