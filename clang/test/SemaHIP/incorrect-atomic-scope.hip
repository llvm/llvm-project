// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx1250 -x hip -emit-llvm -fcuda-is-device %s -o - | FileCheck %s

#define __device__ __attribute__((device))

// Both atomics produce the wrong scope in LLVM IR because a HIP scope was
// incorrectly passed where a Clang scope was expected. But no error or warning
// is generated.
//
// CHECK-LABEL: test_builtin_rmw
// CHECK: atomicrmw fmax {{.*}} syncscope("singlethread")
//
// CHECK-LABEL: test_scoped_atomic
// CHECK: atomicrmw {{.*}} syncscope("singlethread")
//
// CHECK-LABEL: test_intrinsic_metadata
// CHECK: call i32 @llvm.amdgcn.flat.load.monitor{{.*}} metadata [[SCOPE:![0-9]+]]
// CHECK: [[SCOPE]] = !{!"wavefront"}

__device__ void test_builtin_rmw(__attribute__((address_space(3))) float *out, float src) {
  *out = __builtin_amdgcn_ds_fmaxf(out, src, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT,  false);
}

__device__ void test_scoped_atomic(int *ptr) {

  __scoped_atomic_fetch_add(ptr, 1, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ int test_intrinsic_metadata(int* ptr)
{
 return __builtin_amdgcn_flat_load_monitor_b32(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
