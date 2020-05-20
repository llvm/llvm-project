// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 %s -emit-llvm -O0 -o - \
// RUN:   -triple=amdgcn-amd-amdhsa  | opt -S | FileCheck %s

void test_memory_fence_success() {
  // CHECK-LABEL: test_memory_fence_success

  // CHECK: fence syncscope("workgroup") seq_cst
  __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "workgroup");

  // CHECK: fence syncscope("agent") acquire
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

  // CHECK: fence seq_cst
  __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "");

  // CHECK: fence syncscope("agent") acq_rel
  __builtin_amdgcn_fence(4, "agent");

  // CHECK: fence syncscope("workgroup") release
  __builtin_amdgcn_fence(3, "workgroup");
}
