// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgpu13.10-unknown-unknown -verify -S -o - %s

typedef unsigned int uint;

void test_wave_match_b32(global uint* out, uint src0, uint src1) {
  *out = __builtin_amdgcn_wave_match_b32(src0); // expected-error {{too few arguments to function call, expected 2, have 1}}
  *out = __builtin_amdgcn_wave_match_b32(src0, src1, src0); // expected-error {{too many arguments to function call, expected 2, have 3}}
}
