// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -x hip %s -emit-llvm -debug-info-kind=limited -o - | FileCheck %s

#include "Inputs/cuda.h"

// A static __device__ local ODR-used by host code is emitted outside any function.
void use(int *);
void fun1() {
  static __device__ int static_var1 = 3;
  use(&static_var1);
}

// CHECK: @_ZZ4fun1vE11static_var1 = addrspace(1) externally_initialized global i32 3
// CHECK: !DIGlobalVariable(name: "static_var1", scope: ![[CU:[0-9]+]]
// CHECK: ![[CU]] = distinct !DICompileUnit(
