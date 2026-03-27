// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx942 -x hip -emit-llvm -fcuda-is-device %s -o - | FileCheck %s

#define __device__ __attribute__((device))

// CHECK: atomicrmw {{.*}} syncscope("singlethread")

__device__ void test(__attribute__((address_space(3))) float *out, float src) {
  *out = __builtin_amdgcn_ds_fmaxf(out, src, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT,  false); // produces the wrong scope, and there is no check for it.
}
