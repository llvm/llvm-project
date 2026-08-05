// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-- -target-cpu tahiti -verify -S -o - %s
// RUN: %clang_cc1 -triple amdgcn-- -target-cpu hawaii -verify -S -o - %s
// RUN: %clang_cc1 -triple amdgcn-- -target-cpu fiji -verify -S -o - %s
// RUN: %clang_cc1 -triple amdgcn-- -target-cpu gfx900 -verify -S -o - %s
// RUN: %clang_cc1 -triple amdgcn-- -target-cpu gfx908 -verify -S -o - %s
// RUN: %clang_cc1 -triple amdgcn-- -target-cpu gfx1010 -verify -S -o - %s
// RUN: %clang_cc1 -triple amdgcn-- -target-cpu gfx1030 -verify -S -o - %s
// RUN: %clang_cc1 -triple amdgcn-- -target-cpu gfx1100 -verify -S -o - %s

typedef unsigned int uint;

void test(global uint* out, uint a, uint b, uint c) {
  *out = __builtin_amdgcn_permlane16_var(a, b, c, 1, 1); // expected-error {{'__builtin_amdgcn_permlane16_var' needs target feature gfx12-insts}}
  *out = __builtin_amdgcn_permlanex16_var(a, b, c, 1, 1); // expected-error {{'__builtin_amdgcn_permlanex16_var' needs target feature gfx12-insts}}
}
