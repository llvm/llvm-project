// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-- -target-cpu gfx1200 -verify -S -o - %s

typedef unsigned int uint;
typedef unsigned short int ushort;
typedef short __attribute__((ext_vector_type(2))) short2;
typedef unsigned short __attribute__((ext_vector_type(2))) ushort2;

void test(global int* out, global short2 *s2out, global ushort2 *us2out,
          uint a, uint b, uint c, short2 s2a, short2 s2b, short2 s2c,
          ushort2 us2a, ushort2 us2b, ushort2 us2c) {
  __builtin_amdgcn_s_setprio_inc_wg(1); // expected-error {{'__builtin_amdgcn_s_setprio_inc_wg' needs target feature setprio-inc-wg-inst}}
  *out = __builtin_amdgcn_bitop3_b32(a, b, c, 1); // expected-error {{'__builtin_amdgcn_bitop3_b32' needs target feature bitop3-insts}}
  *out = __builtin_amdgcn_bitop3_b16((ushort)a, (ushort)b, (ushort)c, 1); // expected-error {{'__builtin_amdgcn_bitop3_b16' needs target feature bitop3-insts}}
  *out = __builtin_amdgcn_add_max_i32(a, b, c, false); // expected-error {{'__builtin_amdgcn_add_max_i32' needs target feature add-min-max-insts}}
  *out = __builtin_amdgcn_add_max_u32(a, b, c, true); // expected-error {{'__builtin_amdgcn_add_max_u32' needs target feature add-min-max-insts}}
  *out = __builtin_amdgcn_add_min_i32(a, b, c, false); // expected-error {{'__builtin_amdgcn_add_min_i32' needs target feature add-min-max-insts}}
  *out = __builtin_amdgcn_add_min_u32(a, b, c, true); // expected-error {{'__builtin_amdgcn_add_min_u32' needs target feature add-min-max-insts}}
  *s2out = __builtin_amdgcn_pk_add_max_i16(s2a, s2b, s2c, false); // expected-error {{'__builtin_amdgcn_pk_add_max_i16' needs target feature pk-add-min-max-insts}}
  *us2out = __builtin_amdgcn_pk_add_max_u16(us2a, us2b, us2c, true); // expected-error {{'__builtin_amdgcn_pk_add_max_u16' needs target feature pk-add-min-max-insts}}
  *s2out = __builtin_amdgcn_pk_add_min_i16(s2a, s2b, s2c, false); // expected-error {{'__builtin_amdgcn_pk_add_min_i16' needs target feature pk-add-min-max-insts}}
  *us2out = __builtin_amdgcn_pk_add_min_u16(us2a, us2b, us2c, true); // expected-error {{'__builtin_amdgcn_pk_add_min_u16' needs target feature pk-add-min-max-insts}}
}
