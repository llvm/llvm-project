// RUN: %clang_cc1 -no-opaque-pointers %s -triple "spir64-unknown-unknown" -emit-llvm -o - | FileCheck %s

// CHECK: target triple = "spir64-unknown-unknown"

typedef struct {
  char c;
  void *v;
  void *v2;
} my_st;

kernel void foo(global long *arg) {
  int res1[sizeof(my_st)  == 24 ? 1 : -1];
  int res2[sizeof(void *) ==  8 ? 1 : -1];
  int res3[sizeof(arg)    ==  8 ? 1 : -1];

  my_st *tmp = 0;
  arg[3] = (long)(&tmp->v);
//CHECK: store i64 8, i64 addrspace(1)*
  arg[4] = (long)(&tmp->v2);
//CHECK: store i64 16, i64 addrspace(1)*
}
