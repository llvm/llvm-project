// RUN: clang-check -analyze "%s" -- -c 2>&1 | FileCheck %s

// CHECK: Dereference of null pointer
int a(int *x) { if(x){} *x = 47; }
