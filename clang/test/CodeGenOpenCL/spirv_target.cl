// RUN: %clang_cc1 %s -triple "spirv32-unknown-unknown" -verify -emit-llvm -o - | FileCheck %s -check-prefix=SPIRV32
// RUN: %clang_cc1 %s -triple "spirv64-unknown-unknown" -verify -emit-llvm -o - | FileCheck %s -check-prefix=SPIRV64

// SPIRV32: target triple = "spirv32-unknown-unknown"
// SPIRV64: target triple = "spirv64-unknown-unknown"

typedef struct {
  char c;
  void *v;
  void *v2;
} my_st;

kernel void foo(global long *arg) {
#if __SPIRV32__ == 1
  int res1[sizeof(my_st)  == 12 ? 1 : -1]; // expected-no-diagnostics
  int res2[sizeof(void *) ==  4 ? 1 : -1]; // expected-no-diagnostics
  int res3[sizeof(arg)    ==  4 ? 1 : -1]; // expected-no-diagnostics
#elif __SPIRV64__ == 1
  int res1[sizeof(my_st)  == 24 ? 1 : -1]; // expected-no-diagnostics
  int res2[sizeof(void *) ==  8 ? 1 : -1]; // expected-no-diagnostics
  int res3[sizeof(arg)    ==  8 ? 1 : -1]; // expected-no-diagnostics
#endif
  my_st *tmp = 0;

  // SPIRV32: store i64 4, ptr addrspace(1)
  // SPIRV64: store i64 8, ptr addrspace(1)
  arg[0] = (long)(&tmp->v);
  // SPIRV32: store i64 8, ptr addrspace(1)
  // SPIRV64: store i64 16, ptr addrspace(1)
  arg[1] = (long)(&tmp->v2);
}
