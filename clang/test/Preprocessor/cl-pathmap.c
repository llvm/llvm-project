// RUN: %clang_cl -E /pathmap:%p=x:/path-to/ %s | FileCheck %s --check-prefix CHECK-REPRODUCABLE
// CHECK-REPRODUCABLE: filename: "x:\\path-to\\cl-pathmap.c"
// CHECK-REPRODUCABLE-NOT: filename:

//NOTE: currently . and .\ just gets eliminated after mapping 
// RUN: %clang_cl -E /pathmap:%p=. %s | FileCheck %s --check-prefix CHECK-SIMPLE
// CHECK-SIMPLE: filename: "cl-pathmap.c"
// CHECK-SIMPLE-NOT: filename:
filename: __FILE__
