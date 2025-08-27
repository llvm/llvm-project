// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-- -target-cpu gfx1200 -verify -S -o - %s

typedef unsigned int uint;
typedef unsigned short int ushort;

void test(global uint* out, uint a, uint b, uint c) {
  __builtin_amdgcn_s_setprio_inc_wg(1); // expected-error {{'__builtin_amdgcn_s_setprio_inc_wg' needs target feature setprio-inc-wg-inst}}
  *out = __builtin_amdgcn_bitop3_b32(a, b, c, 1); // expected-error {{'__builtin_amdgcn_bitop3_b32' needs target feature bitop3-insts}}
  *out = __builtin_amdgcn_bitop3_b16((ushort)a, (ushort)b, (ushort)c, 1); // expected-error {{'__builtin_amdgcn_bitop3_b16' needs target feature bitop3-insts}}
}
