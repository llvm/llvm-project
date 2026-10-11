// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgpu6.00-- -verify -S -o - %s
// RUN: %clang_cc1 -triple amdgpu7.01-- -verify -S -o - %s
// RUN: %clang_cc1 -triple amdgpu8.03-- -verify -S -o - %s
// RUN: %clang_cc1 -triple amdgpu9.00-- -verify -S -o - %s
// RUN: %clang_cc1 -triple amdgpu9.08-- -verify -S -o - %s
// RUN: %clang_cc1 -triple amdgpu10.10-- -verify -S -o - %s
// RUN: %clang_cc1 -triple amdgpu10.30-- -verify -S -o - %s
// RUN: %clang_cc1 -triple amdgpu11.00-- -verify -S -o - %s

typedef unsigned int uint;

void test(global uint* out, uint a, uint b, uint c) {
  *out = __builtin_amdgcn_permlane16_var(a, b, c, 1, 1); // expected-error {{'__builtin_amdgcn_permlane16_var' needs target feature gfx12-insts}}
  *out = __builtin_amdgcn_permlanex16_var(a, b, c, 1, 1); // expected-error {{'__builtin_amdgcn_permlanex16_var' needs target feature gfx12-insts}}
}
