// RUN: %clang_cc1 -E %s | FileCheck %s --match-full-lines --strict-whitespace
// CHECK: # define X 3

#define H # 
 #define D define 
 
 #define DEFINE(a, b) H D a b 
 
 DEFINE(X, 3) 

