// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -x hip %s \
// RUN:   -std=c++11 -O3 -mllvm -amdgpu-internalize-symbols -emit-llvm -o - \
// RUN:   -target-cpu gfx906 | FileCheck %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -x hip %s \
// RUN:   -std=c++11 -O3 -mllvm -amdgpu-internalize-symbols -emit-llvm -o - \
// RUN:   -target-cpu gfx906 | FileCheck -check-prefix=NEGCHK %s

#include "Inputs/cuda.h"

// AMDGPU internalize unused global variables for whole-program compilation
// (-fno-gpu-rdc for each TU, or -fgpu-rdc for LTO), which are then
// eliminated by global DCE. If there are invisible unused address space casts
// for global variables, these dead users need to be eliminated by global
// DCE before internalization. This test makes sure unused global variables
// are eliminated.

// CHECK-DAG: @v1
__device__ int v1;

// CHECK-DAG: @v2
__constant__ int v2;

// Check unused device/constant variables are eliminated.

// NEGCHK-NOT: @_ZL2v3
constexpr int v3 = 1;

// Check managed variables are always kept.

// CHECK-DAG: @v4
__managed__ int v4;

// Check used device/constant variables are not eliminated.
// CHECK-DAG: @u1
__device__ int u1;

// CHECK-DAG: @u2
__constant__ int u2;

// Check u3 is kept because its address is taken.
// CHECK-DAG: @_ZL2u3
constexpr int u3 = 2;

// Check u4 is not kept because it is not ODR-use.
// NEGCHK-NOT: @_ZL2u4
constexpr int u4 = 3;

__device__ int fun1(const int& x);

__global__ void kern1(int *x) {
  *x = u1 + u2 + fun1(u3) + u4;
}
