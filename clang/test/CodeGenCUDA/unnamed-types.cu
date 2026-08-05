// RUN: %clang_cc1 -std=c++11 -x hip -triple x86_64-linux-gnu -aux-triple amdgcn-amd-amdhsa -emit-llvm %s -o - | FileCheck %s --check-prefix=HOST
// RUN: %clang_cc1 -std=c++11 -x hip -triple x86_64-pc-windows-msvc -aux-triple amdgcn-amd-amdhsa -emit-llvm %s -o - | FileCheck %s --check-prefix=MSVC
// RUN: %clang_cc1 -std=c++11 -x hip -triple amdgcn-amd-amdhsa -fcuda-is-device -emit-llvm %s -o - | FileCheck %s --check-prefix=DEVICE

#include "Inputs/cuda.h"

// HOST: @0 = private unnamed_addr constant [43 x i8] c"_Z2k0IZZ2f1PfENKUlS0_E_clES0_EUlfE_EvS0_T_\00", align 1
// HOST: @1 = private unnamed_addr constant [60 x i8] c"_Z2k1IZ2f1PfEUlfE_Z2f1S0_EUlffE_Z2f1S0_EUlfE0_EvS0_T_T0_T1_\00", align 1
// Check that, on MSVC, the same device kernel mangling name is generated.
// MSVC: @0 = private unnamed_addr constant [43 x i8] c"_Z2k0IZZ2f1PfENKUlS0_E_clES0_EUlfE_EvS0_T_\00", align 1
// MSVC: @1 = private unnamed_addr constant [60 x i8] c"_Z2k1IZ2f1PfEUlfE_Z2f1S0_EUlffE_Z2f1S0_EUlfE0_EvS0_T_T0_T1_\00", align 1

__device__ float d0(float x) {
  return [](float x) { return x + 1.f; }(x);
}

__device__ float d1(float x) {
  return [](float x) { return x * 2.f; }(x);
}

// DEVICE: amdgpu_kernel void @_Z2k0IZZ2f1PfENKUlS0_E_clES0_EUlfE_EvS0_T_(
// DEVICE: define internal noundef float @_ZZZ2f1PfENKUlS_E_clES_ENKUlfE_clEf(
template <typename F>
__global__ void k0(float *p, F f) {
  p[0] = f(p[0]) + d0(p[1]) + d1(p[2]);
}

// DEVICE: amdgpu_kernel void @_Z2k1IZ2f1PfEUlfE_Z2f1S0_EUlffE_Z2f1S0_EUlfE0_EvS0_T_T0_T1_(
// DEVICE: define internal noundef float @_ZZ2f1PfENKUlfE_clEf(
// DEVICE: define internal noundef float @_ZZ2f1PfENKUlffE_clEff(
// DEVICE: define internal noundef float @_ZZ2f1PfENKUlfE0_clEf(
template <typename F0, typename F1, typename F2>
__global__ void k1(float *p, F0 f0, F1 f1, F2 f2) {
  p[0] = f0(p[0]) + f1(p[1], p[2]) + f2(p[3]);
}

void f0(float *p) {
  [](float *p) {
    *p = 1.f;
  }(p);
}

// The inner/outer lambdas are required to be mangled following ODR but their
// linkages are still required to keep the original `internal` linkage.

// HOST: define internal void @_ZZ2f1PfENKUlS_E_clES_(
void f1(float *p) {
  [](float *p) {
    k0<<<1,1>>>(p, [] __device__ (float x) { return x + 3.f; });
  }(p);
  k1<<<1,1>>>(p,
              [] __device__ (float x) { return x + 4.f; },
              [] __device__ (float x, float y) { return x * y; },
              [] __device__ (float x) { return x + 5.f; });
}
// HOST: @__hip_register_globals
// HOST: __hipRegisterFunction{{.*}}@_Z2k0IZZ2f1PfENKUlS0_E_clES0_EUlfE_EvS0_T_{{.*}}@0
// HOST: __hipRegisterFunction{{.*}}@_Z2k1IZ2f1PfEUlfE_Z2f1S0_EUlffE_Z2f1S0_EUlfE0_EvS0_T_T0_T1_{{.*}}@1
// MSVC: __hipRegisterFunction{{.*}}@"??$k0@V<lambda_1>@?0???R1?0??f1@@YAXPEAM@Z@QEBA@0@Z@@@YAXPEAMV<lambda_1>@?0???R0?0??f1@@YAX0@Z@QEBA@0@Z@@Z{{.*}}@0
// MSVC: __hipRegisterFunction{{.*}}@"??$k1@V<lambda_2>@?0??f1@@YAXPEAM@Z@V<lambda_3>@?0??2@YAX0@Z@V<lambda_4>@?0??2@YAX0@Z@@@YAXPEAMV<lambda_2>@?0??f1@@YAX0@Z@V<lambda_3>@?0??1@YAX0@Z@V<lambda_4>@?0??1@YAX0@Z@@Z{{.*}}@1
