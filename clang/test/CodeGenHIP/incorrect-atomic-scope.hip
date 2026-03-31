// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx942 -x hip -emit-llvm -fcuda-is-device %s -o - | FileCheck %s

#define __device__ __attribute__((device))

// Both atomics produce the wrong scope in LLVM IR because a HIP scope was
// incorrectly passed where a Clang scope was expected. But no error or warning
// is generated.
//
// CHECK: atomicrmw fmax {{.*}} syncscope("singlethread")
// CHECK: atomicrmw {{.*}} syncscope("singlethread")

__device__ void test(__attribute__((address_space(3))) float *out, int *ptr, float src) {
  *out = __builtin_amdgcn_ds_fmaxf(out, src, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT,  false);
  __scoped_atomic_fetch_add(ptr, 1, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_AGENT);
}
