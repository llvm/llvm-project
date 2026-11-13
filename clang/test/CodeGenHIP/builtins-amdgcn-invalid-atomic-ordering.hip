// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgpu12.50-amd-amdhsa -fcuda-is-device -emit-llvm -verify -o - %s | FileCheck %s

// Invalid atomic orderings are only diagnosed with a warning and must still be
// able to be codegenned (with the ordering usually replaced by "relaxed" instead)

#define __device__ __attribute__((device))

typedef int v4i __attribute__((ext_vector_type(4)));

// CHECK-LABEL: @test_coop_load_release(
// CHECK: call i32 @llvm.amdgcn.cooperative.atomic.load.32x4B.p0(ptr %{{.*}}, i32 0, metadata
extern "C" __device__ int test_coop_load_release(int *p) {
  return __builtin_amdgcn_cooperative_atomic_load_32x4B(p, __ATOMIC_RELEASE, ""); // expected-warning {{memory order argument to atomic operation is invalid}}
}

// CHECK-LABEL: @test_coop_store_acquire(
// CHECK: call void @llvm.amdgcn.cooperative.atomic.store.8x16B.p0(ptr %{{.*}}, <4 x i32> %{{.*}}, i32 0, metadata
extern "C" __device__ void test_coop_store_acquire(v4i *p, v4i v) {
  __builtin_amdgcn_cooperative_atomic_store_8x16B(p, v, __ATOMIC_ACQUIRE, ""); // expected-warning {{memory order argument to atomic operation is invalid}}
}

// CHECK-LABEL: @test_load_monitor_acq_rel(
// CHECK: call i32 @llvm.amdgcn.flat.load.monitor.b32.i32(ptr %{{.*}}, i32 0, metadata
extern "C" __device__ int test_load_monitor_acq_rel(int *p) {
  return __builtin_amdgcn_flat_load_monitor_b32(p, __ATOMIC_ACQ_REL, __MEMORY_SCOPE_SYSTEM); // expected-warning {{memory order argument to atomic operation is invalid}}
}

// CHECK-LABEL: @test_load_monitor_out_of_range(
// CHECK: call i32 @llvm.amdgcn.flat.load.monitor.b32.i32(ptr %{{.*}}, i32 0, metadata
extern "C" __device__ int test_load_monitor_out_of_range(int *p) {
  return __builtin_amdgcn_flat_load_monitor_b32(p, 42, __MEMORY_SCOPE_SYSTEM); // expected-warning {{memory order argument to atomic operation is invalid}}
}

// CHECK-LABEL: @test_atomic_inc_out_of_range(
// CHECK: atomicrmw uinc_wrap ptr %{{.*}}, i32 %{{.*}} syncscope("workgroup") monotonic
extern "C" __device__ unsigned test_atomic_inc_out_of_range(unsigned *p, unsigned v) {
  return __builtin_amdgcn_atomic_inc32(p, v, 42, "workgroup"); // expected-warning {{memory order argument to atomic operation is invalid}}
}

// CHECK-LABEL: @test_fence_consume(
// CHECK: fence syncscope("workgroup") acquire
extern "C" __device__ void test_fence_consume() {
  __builtin_amdgcn_fence(__ATOMIC_CONSUME, "workgroup"); // expected-warning {{memory order argument to atomic operation is invalid}}
}

// CHECK-LABEL: @test_fence_relaxed(
// CHECK-NOT: fence
// CHECK: ret void
extern "C" __device__ void test_fence_relaxed() {
  __builtin_amdgcn_fence(__ATOMIC_RELAXED, "workgroup"); // expected-warning {{memory order argument to atomic operation is invalid}}
}

// CHECK-LABEL: @test_fence_out_of_range(
// CHECK-NOT: fence
// CHECK: ret void
extern "C" __device__ void test_fence_out_of_range() {
  __builtin_amdgcn_fence(42, "workgroup"); // expected-warning {{memory order argument to atomic operation is invalid}}
}
