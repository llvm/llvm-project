// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 %s -verify -emit-llvm -O0 -o - \
// RUN:   -triple=amdgcn-amd-amdhsa \
// RUN:   -target-cpu gfx942

// Test that [[clang::amdgpu_av("none")]] on AMDGPU-specific atomic builtins
// is rejected with a warning.

void test_atomic_inc32_av(volatile unsigned *p, unsigned val) {
  [[clang::amdgpu_av("none")]] __builtin_amdgcn_atomic_inc32(p, val, __ATOMIC_SEQ_CST, "agent"); // expected-warning {{only applies to atomic expressions or Clang atomic builtins}}
}

void test_atomic_dec64_av(volatile unsigned long long *p, unsigned long long val) {
  [[clang::amdgpu_av("none")]] __builtin_amdgcn_atomic_dec64(p, val, __ATOMIC_ACQUIRE, "agent"); // expected-warning {{only applies to atomic expressions or Clang atomic builtins}}
}

void test_ds_faddf_av(float __attribute__((address_space(3))) *p, float val) {
  [[clang::amdgpu_av("none")]] __builtin_amdgcn_ds_faddf(p, val, __ATOMIC_ACQUIRE, __ATOMIC_ACQUIRE, false); // expected-warning {{only applies to atomic expressions or Clang atomic builtins}}
}

void test_global_atomic_fadd_av(float __attribute__((address_space(1))) *p, float val) {
  [[clang::amdgpu_av("none")]] __builtin_amdgcn_global_atomic_fadd_f32(p, val); // expected-warning {{only applies to atomic expressions or Clang atomic builtins}}
}
