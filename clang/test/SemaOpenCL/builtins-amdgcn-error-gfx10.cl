// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-- -target-cpu tahiti -verify -S -o - %s
// RUN: %clang_cc1 -triple amdgcn-- -target-cpu hawaii -verify -S -o - %s
// RUN: %clang_cc1 -triple amdgcn-- -target-cpu fiji -verify -S -o - %s
// RUN: %clang_cc1 -triple amdgcn-- -target-cpu gfx900 -verify -S -o - %s
// RUN: %clang_cc1 -triple amdgcn-- -target-cpu gfx908 -verify -S -o - %s

typedef unsigned int uint;


void test(global uint* out, uint a, uint b, uint c, uint d) {
  *out = __builtin_amdgcn_permlane16(a, b, c, d, 1, 1); // expected-error {{'__builtin_amdgcn_permlane16' needs target feature gfx10-insts}}
  *out = __builtin_amdgcn_permlanex16(a, b, c, d, 1, 1);  // expected-error {{'__builtin_amdgcn_permlanex16' needs target feature gfx10-insts}}
  *out = __builtin_amdgcn_mov_dpp8(a, 1);  // expected-error {{'__builtin_amdgcn_mov_dpp8' needs target feature gfx10-insts}}
}
