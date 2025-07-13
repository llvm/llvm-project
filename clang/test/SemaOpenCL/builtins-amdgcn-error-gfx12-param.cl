// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-- -target-cpu gfx1200 -verify -S -o - %s

typedef unsigned int uint;

void test_permlane16_var(global uint* out, uint a, uint b, uint c, uint d) {
  *out = __builtin_amdgcn_permlane16_var(a, b, c, d, 1); // expected-error{{argument to '__builtin_amdgcn_permlane16_var' must be a constant integer}}
  *out = __builtin_amdgcn_permlane16_var(a, b, c, 1, d); // expected-error{{argument to '__builtin_amdgcn_permlane16_var' must be a constant integer}}
}

void test_permlanex16_var(global uint* out, uint a, uint b, uint c, uint d) {
  *out = __builtin_amdgcn_permlanex16_var(a, b, c, d, 1); // expected-error{{argument to '__builtin_amdgcn_permlanex16_var' must be a constant integer}}
  *out = __builtin_amdgcn_permlanex16_var(a, b, c, 1, d); // expected-error{{argument to '__builtin_amdgcn_permlanex16_var' must be a constant integer}}
}
