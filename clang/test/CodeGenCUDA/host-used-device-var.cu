// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -x hip %s \
// RUN:   -std=c++11 -O3 -mllvm -amdgpu-internalize-symbols -emit-llvm -o - \
// RUN:   | FileCheck %s

#include "Inputs/cuda.h"

// Check device variables used by neither host nor device functioins are not kept.

// CHECK-NOT: @v1
__device__ int v1;

// CHECK-NOT: @v2
__constant__ int v2;

// CHECK-NOT: @_ZL2v3
static __device__ int v3;

// Check device variables used by host functions are kept.

// CHECK-DAG: @u1
__device__ int u1;

// CHECK-DAG: @u2
__constant__ int u2;

// Check host-used static device var is in llvm.compiler.used.
// CHECK-DAG: @_ZL2u3
static __device__ int u3;

// Check device-used static device var is emitted but is not in llvm.compiler.used.
// CHECK-DAG: @_ZL2u4
static __device__ int u4;

// Check device variables with used attribute are always kept.
// CHECK-DAG: @u5
__device__ __attribute__((used)) int u5;

int fun1() {
  return u1 + u2 + u3;
}

__global__ void kern1(int **x) {
  *x = &u4;
}
// Check the exact list of variables to ensure @_ZL2u4 is not among them.
// CHECK: @llvm.compiler.used = {{[^@]*}} @_ZL2u3 {{[^@]*}} @u1 {{[^@]*}} @u2 {{[^@]*}} @u5
