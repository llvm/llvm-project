// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-- -verify -S -o - %s

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__attribute__((target("arch=tahiti")))
void test_f16_tahiti(global half *out, half a, half b, half c)
{
  *out = __builtin_amdgcn_div_fixuph(a, b, c); // expected-error {{'__builtin_amdgcn_div_fixuph' needs target feature 16-bit-insts}}
  *out = __builtin_amdgcn_rcph(a); // expected-error {{'__builtin_amdgcn_rcph' needs target feature 16-bit-insts}}
  *out = __builtin_amdgcn_rsqh(a); // expected-error {{'__builtin_amdgcn_rsqh' needs target feature 16-bit-insts}}
  *out = __builtin_amdgcn_sinh(a); // expected-error {{'__builtin_amdgcn_sinh' needs target feature 16-bit-insts}}
  *out = __builtin_amdgcn_cosh(a); // expected-error {{'__builtin_amdgcn_cosh' needs target feature 16-bit-insts}}
  *out = __builtin_amdgcn_ldexph(a, b); // expected-error {{'__builtin_amdgcn_ldexph' needs target feature 16-bit-insts}}
  *out = __builtin_amdgcn_frexp_manth(a); // expected-error {{'__builtin_amdgcn_frexp_manth' needs target feature 16-bit-insts}}
  *out = __builtin_amdgcn_frexp_exph(a); // expected-error {{'__builtin_amdgcn_frexp_exph' needs target feature 16-bit-insts}}
  *out = __builtin_amdgcn_fracth(a); // expected-error {{'__builtin_amdgcn_fracth' needs target feature 16-bit-insts}}
  *out = __builtin_amdgcn_classh(a, b); // expected-error {{'__builtin_amdgcn_classh' needs target feature 16-bit-insts}}
  *out = __builtin_amdgcn_fmed3h(a, b, c); // expected-error {{'__builtin_amdgcn_fmed3h' needs target feature gfx9-insts}}
}
