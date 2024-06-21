// See also: clang/test/CodeGenOpenCL/spirv_target.cl
// RUN: %clang_cc1 -cl-std=CL3.0 -fclangir -emit-cir -triple spirv64-unknown-unknown %s -o %t_64.cir
// RUN: FileCheck --input-file=%t_64.cir %s --check-prefix=CIR-SPIRV64
// RUN: %clang_cc1 -cl-std=CL3.0 -fclangir -emit-llvm -triple spirv64-unknown-unknown %s -o %t_64.ll
// RUN: FileCheck --input-file=%t_64.ll %s --check-prefix=LLVM-SPIRV64

// CIR-SPIRV64: cir.triple = "spirv64-unknown-unknown"
// LLVM-SPIRV64: target triple = "spirv64-unknown-unknown"

typedef struct {
  char c;
  void *v;
  void *v2;
} my_st;

// CIR-SPIRV64: cir.func @func(
// LLVM-SPIRV64: @func(
kernel void func(global long *arg) {
  int res1[sizeof(my_st)  == 24 ? 1 : -1]; // expected-no-diagnostics
  int res2[sizeof(void *) ==  8 ? 1 : -1]; // expected-no-diagnostics
  int res3[sizeof(arg)    ==  8 ? 1 : -1]; // expected-no-diagnostics

  my_st *tmp = 0;

  // LLVM-SPIRV64: store i64 8, ptr addrspace(1)
  arg[0] = (long)(&tmp->v);
  // LLVM-SPIRV64: store i64 16, ptr addrspace(1)
  arg[1] = (long)(&tmp->v2);
}
