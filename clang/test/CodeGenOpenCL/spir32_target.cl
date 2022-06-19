// RUN: %clang_cc1 -no-opaque-pointers %s -triple "spir-unknown-unknown" -emit-llvm -o - | FileCheck %s

// CHECK: target triple = "spir-unknown-unknown"

typedef struct {
  char c;
  void *v;
  void *v2;
} my_st;

kernel void foo(global long *arg) {
  int res1[sizeof(my_st)  == 12 ? 1 : -1];
  int res2[sizeof(void *) ==  4 ? 1 : -1];
  int res3[sizeof(arg)    ==  4 ? 1 : -1];

  my_st *tmp = 0;

  arg[0] = (long)(&tmp->v);
//CHECK: store i64 4, i64 addrspace(1)*
  arg[1] = (long)(&tmp->v2);
//CHECK: store i64 8, i64 addrspace(1)*
}
