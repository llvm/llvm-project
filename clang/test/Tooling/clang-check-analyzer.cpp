// RUN: clang-check -analyze "%s" -- -c 2>&1 | FileCheck %s

// CHECK: Dereference of null pointer
void a(int *x) { if(x){} *x = 47; }
