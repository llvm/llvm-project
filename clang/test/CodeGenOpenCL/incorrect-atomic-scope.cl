// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx900 -emit-llvm -o - %s | FileCheck --check-prefixes=CHECK %s
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -emit-llvm -o - %s | FileCheck --check-prefixes=CHECK %s

// Both atomics produce the wrong scope in LLVM IR because a HIP scope was
// incorrectly passed where a Clang scope was expected. But no error or warning
// is generated.
//
// CHECK: atomicrmw fmax {{.*}} syncscope("workgroup")
// CHECK: atomicrmw {{.*}} syncscope("workgroup")

#if !defined(__SPIRV__)
void test(local float *out, int *ptr, float src) {
#else
void test(__attribute__((address_space(3))) float *out, int *ptr, float src) {
#endif
  *out = __builtin_amdgcn_ds_fmaxf(out, src, __ATOMIC_RELAXED, __OPENCL_MEMORY_SCOPE_DEVICE,  false);
  __scoped_atomic_fetch_add(ptr, 1, __ATOMIC_SEQ_CST, __OPENCL_MEMORY_SCOPE_DEVICE);
}
