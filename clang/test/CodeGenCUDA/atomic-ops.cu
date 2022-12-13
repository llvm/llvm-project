// RUN: %clang_cc1 -x hip -std=c++11 -triple amdgcn -fcuda-is-device -emit-llvm %s -o - | FileCheck %s
#include "Inputs/cuda.h"

// CHECK-LABEL: @_Z24atomic32_op_singlethreadPiii
// CHECK: cmpxchg ptr {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}} syncscope("singlethread-one-as")
// CHECK: cmpxchg weak ptr {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}} syncscope("singlethread-one-as") monotonic monotonic, align 4
// CHECK: atomicrmw xchg ptr {{%[0-9]+}}, i32 {{%[0-9]+}} syncscope("singlethread-one-as")
// CHECK: atomicrmw add ptr {{%[0-9]+}}, i32 {{%[0-9]+}} syncscope("singlethread-one-as")
// CHECK: atomicrmw and ptr {{%[0-9]+}}, i32 {{%[0-9]+}} syncscope("singlethread-one-as")
// CHECK: atomicrmw or ptr {{%[0-9]+}}, i32 {{%[0-9]+}} syncscope("singlethread-one-as")
// CHECK: atomicrmw xor ptr {{%[0-9]+}}, i32 {{%[0-9]+}} syncscope("singlethread-one-as")
// CHECK: atomicrmw min ptr {{%[0-9]+}}, i32 {{%[0-9]+}} syncscope("singlethread-one-as")
// CHECK: atomicrmw max ptr {{%[0-9]+}}, i32 {{%[0-9]+}} syncscope("singlethread-one-as")
// CHECK: load atomic i32, ptr {{%[0-9]+}} syncscope("singlethread-one-as") monotonic, align 4
// CHECK: store atomic i32 %{{.*}}, ptr %{{.*}} syncscope("singlethread-one-as") monotonic, align 4
__device__ int atomic32_op_singlethread(int *ptr, int val, int desired) {
  bool flag = __hip_atomic_compare_exchange_strong(ptr, &val, desired, __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  flag = __hip_atomic_compare_exchange_weak(ptr, &val, desired, __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  val = __hip_atomic_exchange(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  val = __hip_atomic_fetch_add(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  val = __hip_atomic_fetch_and(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  val = __hip_atomic_fetch_or(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  val = __hip_atomic_fetch_xor(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  val = __hip_atomic_fetch_min(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  val = __hip_atomic_fetch_max(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  val = __hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  __hip_atomic_store(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  return flag ? val : desired;
}

// CHECK-LABEL: @_Z25atomicu32_op_singlethreadPjjj
// CHECK: atomicrmw umin ptr {{%[0-9]+}}, i32 {{%[0-9]+}} syncscope("singlethread-one-as")
// CHECK: atomicrmw umax ptr {{%[0-9]+}}, i32 {{%[0-9]+}} syncscope("singlethread-one-as")
__device__ unsigned int atomicu32_op_singlethread(unsigned int *ptr, unsigned int val, unsigned int desired) {
  val = __hip_atomic_fetch_min(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  val = __hip_atomic_fetch_max(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  return val;
}

// CHECK-LABEL: @_Z21atomic32_op_wavefrontPiii
// CHECK: cmpxchg ptr {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}} syncscope("wavefront-one-as")
// CHECK: cmpxchg weak ptr {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}} syncscope("wavefront-one-as") monotonic monotonic, align 4
// CHECK: atomicrmw xchg ptr {{%[0-9]+}}, i32 {{%[0-9]+}} syncscope("wavefront-one-as")
// CHECK: atomicrmw add ptr {{%[0-9]+}}, i32 {{%[0-9]+}} syncscope("wavefront-one-as")
// CHECK: atomicrmw and ptr {{%[0-9]+}}, i32 {{%[0-9]+}} syncscope("wavefront-one-as")
// CHECK: atomicrmw or ptr {{%[0-9]+}}, i32 {{%[0-9]+}} syncscope("wavefront-one-as")
// CHECK: atomicrmw xor ptr {{%[0-9]+}}, i32 {{%[0-9]+}} syncscope("wavefront-one-as")
// CHECK: atomicrmw min ptr {{%[0-9]+}}, i32 {{%[0-9]+}} syncscope("wavefront-one-as")
// CHECK: atomicrmw max ptr {{%[0-9]+}}, i32 {{%[0-9]+}} syncscope("wavefront-one-as")
// CHECK: load atomic i32, ptr {{%[0-9]+}} syncscope("wavefront-one-as") monotonic, align 4
// CHECK: store atomic i32 %{{.*}}, ptr %{{.*}} syncscope("wavefront-one-as") monotonic, align 4
__device__ int atomic32_op_wavefront(int *ptr, int val, int desired) {
  bool flag = __hip_atomic_compare_exchange_strong(ptr, &val, desired, __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WAVEFRONT);
  flag = __hip_atomic_compare_exchange_weak(ptr, &val, desired, __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WAVEFRONT);
  val = __hip_atomic_exchange(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WAVEFRONT);
  val = __hip_atomic_fetch_add(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WAVEFRONT);
  val = __hip_atomic_fetch_and(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WAVEFRONT);
  val = __hip_atomic_fetch_or(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WAVEFRONT);
  val = __hip_atomic_fetch_xor(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WAVEFRONT);
  val = __hip_atomic_fetch_min(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WAVEFRONT);
  val = __hip_atomic_fetch_max(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WAVEFRONT);
  val = __hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WAVEFRONT);
  __hip_atomic_store(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WAVEFRONT);
  return flag ? val : desired;
}

// CHECK-LABEL: @_Z22atomicu32_op_wavefrontPjjj
// CHECK: atomicrmw umin ptr {{%[0-9]+}}, i32 {{%[0-9]+}} syncscope("wavefront-one-as")
// CHECK: atomicrmw umax ptr {{%[0-9]+}}, i32 {{%[0-9]+}} syncscope("wavefront-one-as")
__device__ unsigned int atomicu32_op_wavefront(unsigned int *ptr, unsigned int val, unsigned int desired) {
  val = __hip_atomic_fetch_min(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WAVEFRONT);
  val = __hip_atomic_fetch_max(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WAVEFRONT);
  return val;
}

// CHECK-LABEL: @_Z21atomic32_op_workgroupPiii
// CHECK: cmpxchg ptr {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}} syncscope("workgroup-one-as")
// CHECK: cmpxchg weak ptr {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}} syncscope("workgroup-one-as") monotonic monotonic, align 4
// CHECK: atomicrmw xchg ptr {{%[0-9]+}}, i32 {{%[0-9]+}} syncscope("workgroup-one-as")
// CHECK: atomicrmw add ptr {{%[0-9]+}}, i32 {{%[0-9]+}} syncscope("workgroup-one-as")
// CHECK: atomicrmw and ptr {{%[0-9]+}}, i32 {{%[0-9]+}} syncscope("workgroup-one-as")
// CHECK: atomicrmw or ptr {{%[0-9]+}}, i32 {{%[0-9]+}} syncscope("workgroup-one-as")
// CHECK: atomicrmw xor ptr {{%[0-9]+}}, i32 {{%[0-9]+}} syncscope("workgroup-one-as")
// CHECK: atomicrmw min ptr {{%[0-9]+}}, i32 {{%[0-9]+}} syncscope("workgroup-one-as")
// CHECK: atomicrmw max ptr {{%[0-9]+}}, i32 {{%[0-9]+}} syncscope("workgroup-one-as")
// CHECK: store atomic i32 %{{.*}}, ptr %{{.*}} syncscope("workgroup-one-as") monotonic, align 4
__device__ int atomic32_op_workgroup(int *ptr, int val, int desired) {
  bool flag = __hip_atomic_compare_exchange_strong(ptr, &val, desired, __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
  flag = __hip_atomic_compare_exchange_weak(ptr, &val, desired, __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
  val = __hip_atomic_exchange(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
  val = __hip_atomic_fetch_add(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
  val = __hip_atomic_fetch_and(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
  val = __hip_atomic_fetch_or(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
  val = __hip_atomic_fetch_xor(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
  val = __hip_atomic_fetch_min(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
  val = __hip_atomic_fetch_max(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
  __hip_atomic_store(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
  return flag ? val : desired;
}

// CHECK-LABEL: @_Z22atomicu32_op_workgroupPjjj
// CHECK: atomicrmw umin ptr {{%[0-9]+}}, i32 {{%[0-9]+}} syncscope("workgroup-one-as")
// CHECK: atomicrmw umax ptr {{%[0-9]+}}, i32 {{%[0-9]+}} syncscope("workgroup-one-as")
__device__ unsigned int atomicu32_op_workgroup(unsigned int *ptr, unsigned int val, unsigned int desired) {
  val = __hip_atomic_fetch_min(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
  val = __hip_atomic_fetch_max(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
  return val;
}

// CHECK-LABEL: @_Z17atomic32_op_agentPiii
// CHECK: cmpxchg ptr {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}} syncscope("agent-one-as")
// CHECK: cmpxchg weak ptr {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}} syncscope("agent-one-as") monotonic monotonic, align 4
// CHECK: atomicrmw xchg ptr {{%[0-9]+}}, i32 {{%[0-9]+}} syncscope("agent-one-as")
// CHECK: atomicrmw add ptr {{%[0-9]+}}, i32 {{%[0-9]+}} syncscope("agent-one-as")
// CHECK: atomicrmw and ptr {{%[0-9]+}}, i32 {{%[0-9]+}} syncscope("agent-one-as")
// CHECK: atomicrmw or ptr {{%[0-9]+}}, i32 {{%[0-9]+}} syncscope("agent-one-as")
// CHECK: atomicrmw xor ptr {{%[0-9]+}}, i32 {{%[0-9]+}} syncscope("agent-one-as")
// CHECK: atomicrmw min ptr {{%[0-9]+}}, i32 {{%[0-9]+}} syncscope("agent-one-as")
// CHECK: atomicrmw max ptr {{%[0-9]+}}, i32 {{%[0-9]+}} syncscope("agent-one-as")
// CHECK: store atomic i32 %{{.*}}, ptr %{{.*}} syncscope("agent-one-as") monotonic, align 4
__device__ int atomic32_op_agent(int *ptr, int val, int desired) {
  bool flag = __hip_atomic_compare_exchange_strong(ptr, &val, desired, __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  flag = __hip_atomic_compare_exchange_weak(ptr, &val, desired, __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  val = __hip_atomic_exchange(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  val = __hip_atomic_fetch_add(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  val = __hip_atomic_fetch_and(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  val = __hip_atomic_fetch_or(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  val = __hip_atomic_fetch_xor(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  val = __hip_atomic_fetch_min(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  val = __hip_atomic_fetch_max(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  __hip_atomic_store(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  return flag ? val : desired;
}

// CHECK-LABEL: @_Z18atomicu32_op_agentPjjj
// CHECK: atomicrmw umin ptr {{%[0-9]+}}, i32 {{%[0-9]+}} syncscope("agent-one-as")
// CHECK: atomicrmw umax ptr {{%[0-9]+}}, i32 {{%[0-9]+}} syncscope("agent-one-as")
__device__ unsigned int atomicu32_op_agent(unsigned int *ptr, unsigned int val, unsigned int desired) {
  val = __hip_atomic_fetch_min(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  val = __hip_atomic_fetch_max(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  return val;
}

// CHECK-LABEL: @_Z18atomic32_op_systemPiii
// CHECK: cmpxchg ptr {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}} syncscope("one-as")
// CHECK: cmpxchg weak ptr {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}} syncscope("one-as") monotonic monotonic, align 4
// CHECK: atomicrmw xchg ptr {{%[0-9]+}}, i32 {{%[0-9]+}} syncscope("one-as")
// CHECK: atomicrmw add ptr {{%[0-9]+}}, i32 {{%[0-9]+}} syncscope("one-as")
// CHECK: atomicrmw and ptr {{%[0-9]+}}, i32 {{%[0-9]+}} syncscope("one-as")
// CHECK: atomicrmw or ptr {{%[0-9]+}}, i32 {{%[0-9]+}} syncscope("one-as")
// CHECK: atomicrmw xor ptr {{%[0-9]+}}, i32 {{%[0-9]+}} syncscope("one-as")
// CHECK: atomicrmw min ptr {{%[0-9]+}}, i32 {{%[0-9]+}} syncscope("one-as")
// CHECK: atomicrmw max ptr {{%[0-9]+}}, i32 {{%[0-9]+}} syncscope("one-as")
// CHECK: load i32, ptr %{{.*}}, align 4
// CHECK: store atomic i32 %{{.*}}, ptr %{{.*}} syncscope("one-as") monotonic, align 4
__device__ int atomic32_op_system(int *ptr, int val, int desired) {
  bool flag = __hip_atomic_compare_exchange_strong(ptr, &val, desired, __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  flag = __hip_atomic_compare_exchange_weak(ptr, &val, desired, __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  val = __hip_atomic_exchange(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  val = __hip_atomic_fetch_add(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  val = __hip_atomic_fetch_and(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  val = __hip_atomic_fetch_or(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  val = __hip_atomic_fetch_xor(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  val = __hip_atomic_fetch_min(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  val = __hip_atomic_fetch_max(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  val = __hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  __hip_atomic_store(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  return flag ? val : desired;
}

// CHECK-LABEL: @_Z19atomicu32_op_systemPjjj
// CHECK: atomicrmw umin ptr {{%[0-9]+}}, i32 {{%[0-9]+}} syncscope("one-as")
// CHECK: atomicrmw umax ptr {{%[0-9]+}}, i32 {{%[0-9]+}} syncscope("one-as")
__device__ unsigned int atomicu32_op_system(unsigned int *ptr, unsigned int val, unsigned int desired) {
  val = __hip_atomic_fetch_min(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  val = __hip_atomic_fetch_max(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  return val;
}

// CHECK-LABEL: @_Z24atomic64_op_singlethreadPxS_xx
// CHECK: cmpxchg ptr {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}} syncscope("singlethread-one-as")
// CHECK: cmpxchg weak ptr {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}} syncscope("singlethread-one-as") monotonic monotonic, align 8
// CHECK: atomicrmw xchg ptr {{%[0-9]+}}, i64 {{%[0-9]+}} syncscope("singlethread-one-as")
// CHECK: atomicrmw add ptr {{%[0-9]+}}, i64 {{%[0-9]+}} syncscope("singlethread-one-as")
// CHECK: atomicrmw and ptr {{%[0-9]+}}, i64 {{%[0-9]+}} syncscope("singlethread-one-as")
// CHECK: atomicrmw or ptr {{%[0-9]+}}, i64 {{%[0-9]+}} syncscope("singlethread-one-as")
// CHECK: atomicrmw xor ptr {{%[0-9]+}}, i64 {{%[0-9]+}} syncscope("singlethread-one-as")
// CHECK: atomicrmw min ptr {{%[0-9]+}}, i64 {{%[0-9]+}} syncscope("singlethread-one-as")
// CHECK: atomicrmw max ptr {{%[0-9]+}}, i64 {{%[0-9]+}} syncscope("singlethread-one-as")
// CHECK: store atomic i64 %{{.*}}, ptr %{{.*}} syncscope("singlethread-one-as") monotonic, align 8
__device__ long long atomic64_op_singlethread(long long *ptr, long long *ptr2, long long val, long long desired) {
  bool flag = __hip_atomic_compare_exchange_strong(ptr, &val, desired, __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  flag = __hip_atomic_compare_exchange_weak(ptr, &val, desired, __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  val = __hip_atomic_exchange(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  val = __hip_atomic_fetch_add(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  val = __hip_atomic_fetch_and(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  val = __hip_atomic_fetch_or(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  val = __hip_atomic_fetch_xor(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  val = __hip_atomic_fetch_min(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  val = __hip_atomic_fetch_max(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  __hip_atomic_store(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  return flag ? val : desired;
}

// CHECK-LABEL: @_Z25atomicu64_op_singlethreadPyS_yy
// CHECK: atomicrmw umin ptr {{%[0-9]+}}, i64 {{%[0-9]+}} syncscope("singlethread-one-as")
// CHECK: atomicrmw umax ptr {{%[0-9]+}}, i64 {{%[0-9]+}} syncscope("singlethread-one-as")
// CHECK: load atomic i64, ptr %{{.*}} syncscope("singlethread-one-as") monotonic, align 8
// CHECK: store atomic i64 %{{.*}}, ptr %{{.*}} syncscope("singlethread-one-as") monotonic, align 8
__device__ unsigned long long atomicu64_op_singlethread(unsigned long long *ptr, unsigned long long *ptr2, unsigned long long val, unsigned long long desired) {
  val = __hip_atomic_fetch_min(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  val = __hip_atomic_fetch_max(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  val = __hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  __hip_atomic_store(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  return val;
}

// CHECK-LABEL: @_Z21atomic64_op_wavefrontPxS_xx
// CHECK: cmpxchg ptr {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}} syncscope("wavefront-one-as")
// CHECK: cmpxchg weak ptr {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}} syncscope("wavefront-one-as") monotonic monotonic, align 8
// CHECK: atomicrmw xchg ptr {{%[0-9]+}}, i64 {{%[0-9]+}} syncscope("wavefront-one-as")
// CHECK: atomicrmw add ptr {{%[0-9]+}}, i64 {{%[0-9]+}} syncscope("wavefront-one-as")
// CHECK: atomicrmw and ptr {{%[0-9]+}}, i64 {{%[0-9]+}} syncscope("wavefront-one-as")
// CHECK: atomicrmw or ptr {{%[0-9]+}}, i64 {{%[0-9]+}} syncscope("wavefront-one-as")
// CHECK: atomicrmw xor ptr {{%[0-9]+}}, i64 {{%[0-9]+}} syncscope("wavefront-one-as")
// CHECK: atomicrmw min ptr {{%[0-9]+}}, i64 {{%[0-9]+}} syncscope("wavefront-one-as")
// CHECK: atomicrmw max ptr {{%[0-9]+}}, i64 {{%[0-9]+}} syncscope("wavefront-one-as")
// CHECK: load atomic i64, ptr {{%[0-9]+}} syncscope("wavefront-one-as") monotonic, align 8
// CHECK: store atomic i64 %{{.*}}, ptr %{{.*}} syncscope("wavefront-one-as") monotonic, align 8
__device__ long long atomic64_op_wavefront(long long *ptr, long long *ptr2, long long val, long long desired) {
  bool flag = __hip_atomic_compare_exchange_strong(ptr, &val, desired, __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WAVEFRONT);
  flag = __hip_atomic_compare_exchange_weak(ptr, &val, desired, __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WAVEFRONT);
  val = __hip_atomic_exchange(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WAVEFRONT);
  val = __hip_atomic_fetch_add(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WAVEFRONT);
  val = __hip_atomic_fetch_and(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WAVEFRONT);
  val = __hip_atomic_fetch_or(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WAVEFRONT);
  val = __hip_atomic_fetch_xor(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WAVEFRONT);
  val = __hip_atomic_fetch_min(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WAVEFRONT);
  val = __hip_atomic_fetch_max(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WAVEFRONT);
  val = __hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WAVEFRONT);
  __hip_atomic_store(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WAVEFRONT);
  return flag ? val : desired;
}

// CHECK-LABEL: @_Z22atomicu64_op_wavefrontPyS_yy
// CHECK: atomicrmw umin ptr {{%[0-9]+}}, i64 {{%[0-9]+}} syncscope("wavefront-one-as")
// CHECK: atomicrmw umax ptr {{%[0-9]+}}, i64 {{%[0-9]+}} syncscope("wavefront-one-as")
// CHECK: load atomic i64, ptr {{%[0-9]+}} syncscope("wavefront-one-as") monotonic, align 8
// CHECK: store atomic i64 %{{.*}}, ptr %{{.*}} syncscope("wavefront-one-as") monotonic, align 8
__device__ unsigned long long atomicu64_op_wavefront(unsigned long long *ptr, unsigned long long *ptr2, unsigned long long val, unsigned long long desired) {
  val = __hip_atomic_fetch_min(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WAVEFRONT);
  val = __hip_atomic_fetch_max(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WAVEFRONT);
  val = __hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WAVEFRONT);
  __hip_atomic_store(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WAVEFRONT);
  return val;
}

// CHECK-LABEL: @_Z21atomic64_op_workgroupPxS_xx
// CHECK: cmpxchg ptr {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}} syncscope("workgroup-one-as")
// CHECK: cmpxchg weak ptr {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}} syncscope("workgroup-one-as") monotonic monotonic, align 8
// CHECK: atomicrmw xchg ptr {{%[0-9]+}}, i64 {{%[0-9]+}} syncscope("workgroup-one-as")
// CHECK: atomicrmw add ptr {{%[0-9]+}}, i64 {{%[0-9]+}} syncscope("workgroup-one-as")
// CHECK: atomicrmw and ptr {{%[0-9]+}}, i64 {{%[0-9]+}} syncscope("workgroup-one-as")
// CHECK: atomicrmw or ptr {{%[0-9]+}}, i64 {{%[0-9]+}} syncscope("workgroup-one-as")
// CHECK: atomicrmw xor ptr {{%[0-9]+}}, i64 {{%[0-9]+}} syncscope("workgroup-one-as")
// CHECK: atomicrmw min ptr {{%[0-9]+}}, i64 {{%[0-9]+}} syncscope("workgroup-one-as")
// CHECK: atomicrmw max ptr {{%[0-9]+}}, i64 {{%[0-9]+}} syncscope("workgroup-one-as")
// CHECK: store atomic i64 %{{.*}}, ptr %{{.*}} syncscope("workgroup-one-as") monotonic, align 8
__device__ long long atomic64_op_workgroup(long long *ptr, long long *ptr2, long long val, long long desired) {
  bool flag = __hip_atomic_compare_exchange_strong(ptr, &val, desired, __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
  flag = __hip_atomic_compare_exchange_weak(ptr, &val, desired, __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
  val = __hip_atomic_exchange(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
  val = __hip_atomic_fetch_add(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
  val = __hip_atomic_fetch_and(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
  val = __hip_atomic_fetch_or(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
  val = __hip_atomic_fetch_xor(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
  val = __hip_atomic_fetch_min(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
  val = __hip_atomic_fetch_max(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
  __hip_atomic_store(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
  return flag ? val : desired;
}

// CHECK-LABEL: @_Z22atomicu64_op_workgroupPyS_yy
// CHECK: atomicrmw umin ptr {{%[0-9]+}}, i64 {{%[0-9]+}} syncscope("workgroup-one-as")
// CHECK: atomicrmw umax ptr {{%[0-9]+}}, i64 {{%[0-9]+}} syncscope("workgroup-one-as")
// CHECK: store atomic i64 %{{.*}}, ptr %{{.*}} syncscope("workgroup-one-as") monotonic, align 8
__device__ unsigned long long atomicu64_op_workgroup(unsigned long long *ptr, unsigned long long *ptr2, unsigned long long val, unsigned long long desired) {
  val = __hip_atomic_fetch_min(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
  val = __hip_atomic_fetch_max(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
  __hip_atomic_store(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
  return val;
}

// CHECK-LABEL: @_Z17atomic64_op_agentPxS_xx
// CHECK: cmpxchg ptr {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}} syncscope("agent-one-as")
// CHECK: cmpxchg weak ptr {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}} syncscope("agent-one-as") monotonic monotonic, align 8
// CHECK: atomicrmw xchg ptr {{%[0-9]+}}, i64 {{%[0-9]+}} syncscope("agent-one-as")
// CHECK: atomicrmw add ptr {{%[0-9]+}}, i64 {{%[0-9]+}} syncscope("agent-one-as")
// CHECK: atomicrmw and ptr {{%[0-9]+}}, i64 {{%[0-9]+}} syncscope("agent-one-as")
// CHECK: atomicrmw or ptr {{%[0-9]+}}, i64 {{%[0-9]+}} syncscope("agent-one-as")
// CHECK: atomicrmw xor ptr {{%[0-9]+}}, i64 {{%[0-9]+}} syncscope("agent-one-as")
// CHECK: atomicrmw min ptr {{%[0-9]+}}, i64 {{%[0-9]+}} syncscope("agent-one-as")
// CHECK: atomicrmw max ptr {{%[0-9]+}}, i64 {{%[0-9]+}} syncscope("agent-one-as")
// CHECK: store atomic i64 %{{.*}}, ptr %{{.*}} syncscope("agent-one-as") monotonic, align 8
__device__ long long atomic64_op_agent(long long *ptr, long long *ptr2, long long val, long long desired) {
  bool flag = __hip_atomic_compare_exchange_strong(ptr, &val, desired, __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  flag = __hip_atomic_compare_exchange_weak(ptr, &val, desired, __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  val = __hip_atomic_exchange(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  val = __hip_atomic_fetch_add(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  val = __hip_atomic_fetch_and(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  val = __hip_atomic_fetch_or(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  val = __hip_atomic_fetch_xor(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  val = __hip_atomic_fetch_min(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  val = __hip_atomic_fetch_max(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  __hip_atomic_store(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  return flag ? val : desired;
}

// CHECK-LABEL: @_Z18atomicu64_op_agentPyS_yy
// CHECK: atomicrmw umin ptr {{%[0-9]+}}, i64 {{%[0-9]+}} syncscope("agent-one-as")
// CHECK: atomicrmw umax ptr {{%[0-9]+}}, i64 {{%[0-9]+}} syncscope("agent-one-as")
// CHECK: store atomic i64 %{{.*}}, ptr %{{.*}} syncscope("agent-one-as") monotonic, align 8
__device__ unsigned long long atomicu64_op_agent(unsigned long long *ptr, unsigned long long *ptr2, unsigned long long val, unsigned long long desired) {
  val = __hip_atomic_fetch_min(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  val = __hip_atomic_fetch_max(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  __hip_atomic_store(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  return val;
}

// CHECK-LABEL: @_Z18atomic64_op_systemPxS_xx
// CHECK: cmpxchg ptr {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}} syncscope("one-as")
// CHECK: cmpxchg weak ptr {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}} syncscope("one-as") monotonic monotonic, align 8
// CHECK: atomicrmw xchg ptr {{%[0-9]+}}, i64 {{%[0-9]+}} syncscope("one-as")
// CHECK: atomicrmw add ptr {{%[0-9]+}}, i64 {{%[0-9]+}} syncscope("one-as")
// CHECK: atomicrmw and ptr {{%[0-9]+}}, i64 {{%[0-9]+}} syncscope("one-as")
// CHECK: atomicrmw or ptr {{%[0-9]+}}, i64 {{%[0-9]+}} syncscope("one-as")
// CHECK: atomicrmw xor ptr {{%[0-9]+}}, i64 {{%[0-9]+}} syncscope("one-as")
// CHECK: atomicrmw min ptr {{%[0-9]+}}, i64 {{%[0-9]+}} syncscope("one-as")
// CHECK: atomicrmw max ptr {{%[0-9]+}}, i64 {{%[0-9]+}} syncscope("one-as")
// CHECK: load i64, ptr %{{.*}}, align 8
// CHECK: store atomic i64 %{{.*}}, ptr %{{.*}} syncscope("one-as") monotonic, align 8
__device__ long long atomic64_op_system(long long *ptr, long long *ptr2, long long val, long long desired) {
  bool flag = __hip_atomic_compare_exchange_strong(ptr, &val, desired, __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  flag = __hip_atomic_compare_exchange_weak(ptr, &val, desired, __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  val = __hip_atomic_exchange(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  val = __hip_atomic_fetch_add(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  val = __hip_atomic_fetch_and(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  val = __hip_atomic_fetch_or(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  val = __hip_atomic_fetch_xor(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  val = __hip_atomic_fetch_min(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  val = __hip_atomic_fetch_max(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  val = __hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  __hip_atomic_store(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  return flag ? val : desired;
}

// CHECK-LABEL: @_Z19atomicu64_op_systemPyS_yy
// CHECK: atomicrmw umin ptr {{%[0-9]+}}, i64 {{%[0-9]+}} syncscope("one-as")
// CHECK: atomicrmw umax ptr {{%[0-9]+}}, i64 {{%[0-9]+}} syncscope("one-as")
// CHECK: load i64, ptr %{{.*}}, align 8
// CHECK: store atomic i64 %{{.*}}, ptr %{{.*}} syncscope("one-as") monotonic, align 8
__device__ unsigned long long atomicu64_op_system(unsigned long long *ptr, unsigned long long *ptr2, unsigned long long val, unsigned long long desired) {
  val = __hip_atomic_fetch_min(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  val = __hip_atomic_fetch_max(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  val = __hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  __hip_atomic_store(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  return val;
}
