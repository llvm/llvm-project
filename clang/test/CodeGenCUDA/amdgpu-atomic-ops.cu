// RUN: %clang_cc1 %s -emit-llvm -o - -triple=amdgcn-amd-amdhsa \
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
