// RUN: %clang_cc1 -std=c++11 -x hip -triple x86_64-linux-gnu -aux-triple amdgcn-amd-amdhsa -emit-llvm %s -o - | FileCheck %s --check-prefix=HOST
// RUN: %clang_cc1 -std=c++11 -x hip -triple amdgcn-amd-amdhsa -fcuda-is-device -emit-llvm %s -o - | FileCheck %s --check-prefix=DEVICE

#include "Inputs/cuda.h"

// HOST: @0 = private unnamed_addr constant [43 x i8] c"_Z2k0IZZ2f1PfENKUlS0_E_clES0_EUlfE_EvS0_T_\00", align 1

__device__ float d0(float x) {
  return [](float x) { return x + 2.f; }(x);
}

__device__ float d1(float x) {
  return [](float x) { return x * 2.f; }(x);
}

// DEVICE: amdgpu_kernel void @_Z2k0IZZ2f1PfENKUlS0_E_clES0_EUlfE_EvS0_T_(
template <typename F>
__global__ void k0(float *p, F f) {
  p[0] = f(p[0]) + d0(p[1]) + d1(p[2]);
}

void f0(float *p) {
  [](float *p) {
    *p = 1.f;
  }(p);
}

// The inner/outer lambdas are required to be mangled following ODR but their
// linkages are still required to keep the original `internal` linkage.

// HOST: define internal void @_ZZ2f1PfENKUlS_E_clES_(
// DEVICE: define internal float @_ZZZ2f1PfENKUlS_E_clES_ENKUlfE_clEf(
void f1(float *p) {
  [](float *p) {
    k0<<<1,1>>>(p, [] __device__ (float x) { return x + 1.f; });
  }(p);
}
// HOST: @__hip_register_globals
// HOST: __hipRegisterFunction{{.*}}@_Z2k0IZZ2f1PfENKUlS0_E_clES0_EUlfE_EvS0_T_{{.*}}@0
