// REQUIRES: amdgpu-registered-target
// RUN: not %clang_cc1 %s -o - -S -triple=amdgcn-amd-amdhsa 2>&1 | FileCheck %s

void test_amdgcn_fence_failure() {

  // CHECK: error: Unsupported atomic synchronization scope
  __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "foobar");
}
