// Verify that applying the deprecation fix-its for __hip_atomic_* builtins
// produces semantically identical LLVM IR.  Both the original source and the
// fixed source are compiled and checked against the same CHECK lines.
//
// RUN: %clang_cc1 -x hip -std=c++11 -triple amdgcn -fcuda-is-device \
// RUN:   -emit-llvm %s -o - | FileCheck %s
// RUN: cp %s %t
// RUN: %clang_cc1 -x hip -std=c++11 -triple amdgcn -fcuda-is-device \
// RUN:   -fixit %t
// RUN: %clang_cc1 -x hip -std=c++11 -triple amdgcn -fcuda-is-device \
// RUN:   -emit-llvm %t -o - | FileCheck %s

#define __device__ __attribute__((device))

// CHECK-LABEL: @_Z14test_all_fixedPiii
// CHECK: load atomic i32, ptr {{%.*}} syncscope("singlethread") monotonic
// CHECK: store atomic i32 %{{.*}}, ptr %{{.*}} syncscope("wavefront") monotonic
// CHECK: atomicrmw xchg ptr {{%.*}}, i32 {{%.*}} syncscope("workgroup") monotonic
// CHECK: atomicrmw add ptr {{%.*}}, i32 {{%.*}} syncscope("cluster") monotonic
// CHECK: atomicrmw sub ptr {{%.*}}, i32 {{%.*}} syncscope("agent") monotonic
// CHECK: atomicrmw and ptr {{%.*}}, i32 {{%.*}} monotonic
// CHECK: atomicrmw or ptr {{%.*}}, i32 {{%.*}} syncscope("singlethread") monotonic
// CHECK: atomicrmw xor ptr {{%.*}}, i32 {{%.*}} syncscope("wavefront") monotonic
// CHECK: atomicrmw min ptr {{%.*}}, i32 {{%.*}} syncscope("workgroup") monotonic
// CHECK: atomicrmw max ptr {{%.*}}, i32 {{%.*}} syncscope("agent") monotonic
__device__ int test_all_fixed(int *p, int v, int d) {
  v = __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WAVEFRONT);
  v = __hip_atomic_exchange(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
  v = __hip_atomic_fetch_add(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_CLUSTER);
  v = __hip_atomic_fetch_sub(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  v = __hip_atomic_fetch_and(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  v = __hip_atomic_fetch_or(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  v = __hip_atomic_fetch_xor(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WAVEFRONT);
  v = __hip_atomic_fetch_min(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
  v = __hip_atomic_fetch_max(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  return v;
}
