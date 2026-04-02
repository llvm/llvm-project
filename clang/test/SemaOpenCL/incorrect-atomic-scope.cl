// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1250 -emit-llvm -o - %s | FileCheck --check-prefixes=CHECK,AMDGCN %s
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -emit-llvm -o - %s | FileCheck --check-prefixes=CHECK %s

// Both atomics produce the wrong scope in LLVM IR because a HIP scope was
// incorrectly passed where a Clang scope was expected. But no error or warning
// is generated.
//
// CHECK-LABEL: test_builtin_rmw
// CHECK: atomicrmw fmax {{.*}} syncscope("workgroup")
//
// CHECK-LABEL: test_scoped_atomic
// CHECK: atomicrmw {{.*}} syncscope("workgroup")
//
// AMDGCN-LABEL: test_intrinsic_metadata
// AMDGCN: call {{.*}} @llvm.amdgcn.flat.load.monitor{{.*}} metadata [[SCOPE:![0-9]+]]
// AMDGCN: [[SCOPE]] = !{!"workgroup"}

#if !defined(__SPIRV__)
void test_builtin_rmw(local float *out, float src) {
#else
void test_builtin_rmw(__attribute__((address_space(3))) float *out, float src) {
#endif
  *out = __builtin_amdgcn_ds_fmaxf(out, src, __ATOMIC_RELAXED, __OPENCL_MEMORY_SCOPE_DEVICE,  false);
}

void test_scoped_atomic(int *ptr) {
  __scoped_atomic_fetch_add(ptr, 1, __ATOMIC_SEQ_CST, __OPENCL_MEMORY_SCOPE_DEVICE);
}

#if !defined(__SPIRV__)
int test_intrinsic_metadata(int* ptr)
{
 return __builtin_amdgcn_flat_load_monitor_b32(ptr, __ATOMIC_RELAXED, __OPENCL_MEMORY_SCOPE_DEVICE);
}
#endif
