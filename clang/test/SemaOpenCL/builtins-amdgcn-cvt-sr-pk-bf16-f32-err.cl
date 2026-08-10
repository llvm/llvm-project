// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx950 -verify -S -o - %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx942 -verify -S -o - %s

typedef __bf16 bfloat2 __attribute__((ext_vector_type(2)));

// v_cvt_sr_pk_bf16_f32 was only added for gfx1250; it is gated on
// cvt-sr-pk-bf16-f32-inst, which is distinct from bf16-cvt-insts (present on
// gfx950 for the non-stochastic v_cvt_pk_bf16_f32).
void test_cvt_sr_pk_bf16_f32(global bfloat2 *out, float a, float b, unsigned sr) {
  *out = __builtin_amdgcn_cvt_sr_pk_bf16_f32(a, b, sr); // expected-error{{'__builtin_amdgcn_cvt_sr_pk_bf16_f32' needs target feature cvt-sr-pk-bf16-f32-inst}}
}
