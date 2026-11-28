// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -cl-std=CL2.0 -triple amdgpu9.4-amd-amdhsa -fsyntax-only -verify %s

void test_nonconstant(int cpol) {
  __builtin_amdgcn_buffer_inv(cpol); // expected-error {{argument to '__builtin_amdgcn_buffer_inv' must be a constant integer}}
}

void test_invalid_policy(void) {
  // expected-error@+1 {{must be a combination of the sc0 (1) and sc1 (16) cache-policy bits}}
  __builtin_amdgcn_buffer_inv(2);
}
