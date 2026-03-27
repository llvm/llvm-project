// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx900 -emit-llvm -o - %s | FileCheck --check-prefixes=CHECK %s
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -emit-llvm -o - %s | FileCheck --check-prefixes=CHECK %s

// CHECK: atomicrmw {{.*}} syncscope("workgroup")

#if !defined(__SPIRV__)
void test(local float *out, float src) {
#else
void test(__attribute__((address_space(3))) float *out, float src) {
#endif
  *out = __builtin_amdgcn_ds_fmaxf(out, src, __ATOMIC_RELAXED, __OPENCL_MEMORY_SCOPE_DEVICE,  false); // produces the wrong scope, and there is no check for it.
}
