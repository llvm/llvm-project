// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 %s -fsyntax-only -triple=amdgpu-amd-amdhsa -verify

void test_amdgcn_fence_failure() {
  __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "foobar"); // expected-error {{unsupported atomic synchronization scope 'foobar'}}
}
