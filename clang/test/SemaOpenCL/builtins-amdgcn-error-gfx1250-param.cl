// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-- -target-cpu gfx1250 -verify -S -o - %s

typedef unsigned int uint;
typedef unsigned short int ushort;

void test_setprio_inc_wg(short a) {
  __builtin_amdgcn_s_setprio_inc_wg(a); // expected-error {{'__builtin_amdgcn_s_setprio_inc_wg' must be a constant integer}}
}

void test_bitop3_args(global uint* out, uint a, uint b, uint c) {
  *out = __builtin_amdgcn_bitop3_b32(a, b, c, a); // expected-error {{argument to '__builtin_amdgcn_bitop3_b32' must be a constant integer}}
  *out = __builtin_amdgcn_bitop3_b16((ushort)a, (ushort)b, (ushort)c, a); // expected-error {{argument to '__builtin_amdgcn_bitop3_b16' must be a constant integer}}
}
