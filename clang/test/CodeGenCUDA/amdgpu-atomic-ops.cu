// RUN: %clang_cc1 -no-opaque-pointers %s -emit-llvm -o - -triple=amdgcn-amd-amdhsa \
// RUN:   -fcuda-is-device -target-cpu gfx906 -fnative-half-type \
// RUN:   -fnative-half-arguments-and-returns | FileCheck %s

// REQUIRES: amdgpu-registered-target

#include "Inputs/cuda.h"
#include <stdatomic.h>

__device__ float ffp1(float *p) {
  // CHECK-LABEL: @_Z4ffp1Pf
  // CHECK: atomicrmw fadd float* {{.*}} monotonic
  return __atomic_fetch_add(p, 1.0f, memory_order_relaxed);
}

__device__ double ffp2(double *p) {
  // CHECK-LABEL: @_Z4ffp2Pd
  // CHECK: atomicrmw fsub double* {{.*}} monotonic
  return __atomic_fetch_sub(p, 1.0, memory_order_relaxed);
}

// long double is the same as double for amdgcn.
__device__ long double ffp3(long double *p) {
  // CHECK-LABEL: @_Z4ffp3Pe
  // CHECK: atomicrmw fsub double* {{.*}} monotonic
  return __atomic_fetch_sub(p, 1.0L, memory_order_relaxed);
}

__device__ double ffp4(double *p, float f) {
  // CHECK-LABEL: @_Z4ffp4Pdf
  // CHECK: fpext float {{.*}} to double
  // CHECK: atomicrmw fsub double* {{.*}} monotonic
  return __atomic_fetch_sub(p, f, memory_order_relaxed);
}

__device__ double ffp5(double *p, int i) {
  // CHECK-LABEL: @_Z4ffp5Pdi
  // CHECK: sitofp i32 {{.*}} to double
  // CHECK: atomicrmw fsub double* {{.*}} monotonic
  return __atomic_fetch_sub(p, i, memory_order_relaxed);
}
