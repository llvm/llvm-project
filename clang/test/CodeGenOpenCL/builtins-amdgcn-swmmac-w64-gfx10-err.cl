// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1030 \
// RUN:   -verify -S -o - %s

typedef int    v2i   __attribute__((ext_vector_type(2)));
typedef float  v4f   __attribute__((ext_vector_type(4)));
typedef half   v4h   __attribute__((ext_vector_type(4)));
typedef short  v4s   __attribute__((ext_vector_type(4)));
typedef int    v4i   __attribute__((ext_vector_type(4)));
typedef half   v8h   __attribute__((ext_vector_type(8)));
typedef short  v8s   __attribute__((ext_vector_type(8)));

void test_amdgcn_swmmac_w64(global v4f* out4f, global v4h* out4h, global v4s* out4s, global v4i* out4i,
                             v4h a4h, v4s a4s, int ai,
                             v8h b8h, v8s b8s, v2i b2i, int bi,
                             v4f c4f, v4h c4h, v4s c4s, v4i c4i,
                             int index)
{
  *out4f = __builtin_amdgcn_swmmac_f32_16x16x32_f16_w64(a4h, b8h, c4f, index); // expected-error{{'__builtin_amdgcn_swmmac_f32_16x16x32_f16_w64' needs target feature swmmac-gfx1200-insts,wavefrontsize64}}
  *out4f = __builtin_amdgcn_swmmac_f32_16x16x32_bf16_w64(a4s, b8s, c4f, index); // expected-error{{'__builtin_amdgcn_swmmac_f32_16x16x32_bf16_w64' needs target feature swmmac-gfx1200-insts,wavefrontsize64}}
  *out4h = __builtin_amdgcn_swmmac_f16_16x16x32_f16_w64(a4h, b8h, c4h, index); // expected-error{{'__builtin_amdgcn_swmmac_f16_16x16x32_f16_w64' needs target feature swmmac-gfx1200-insts,wavefrontsize64}}
  *out4s = __builtin_amdgcn_swmmac_bf16_16x16x32_bf16_w64(a4s, b8s, c4s, index); // expected-error{{'__builtin_amdgcn_swmmac_bf16_16x16x32_bf16_w64' needs target feature swmmac-gfx1200-insts,wavefrontsize64}}
  *out4i = __builtin_amdgcn_swmmac_i32_16x16x32_iu8_w64(true, ai, true, b2i, c4i, index, true); // expected-error{{'__builtin_amdgcn_swmmac_i32_16x16x32_iu8_w64' needs target feature swmmac-gfx1200-insts,wavefrontsize64}}
  *out4i = __builtin_amdgcn_swmmac_i32_16x16x32_iu4_w64(true, ai, true, bi, c4i, index, true); // expected-error{{'__builtin_amdgcn_swmmac_i32_16x16x32_iu4_w64' needs target feature swmmac-gfx1200-insts,wavefrontsize64}}
  *out4i = __builtin_amdgcn_swmmac_i32_16x16x64_iu4_w64(true, ai, true, b2i, c4i, index, true); // expected-error{{'__builtin_amdgcn_swmmac_i32_16x16x64_iu4_w64' needs target feature swmmac-gfx1200-insts,wavefrontsize64}}
  *out4f = __builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w64(ai, b2i, c4f, index); // expected-error{{'__builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w64' needs target feature swmmac-gfx1200-insts,wavefrontsize64}}
  *out4f = __builtin_amdgcn_swmmac_f32_16x16x32_fp8_bf8_w64(ai, b2i, c4f, index); // expected-error{{'__builtin_amdgcn_swmmac_f32_16x16x32_fp8_bf8_w64' needs target feature swmmac-gfx1200-insts,wavefrontsize64}}
  *out4f = __builtin_amdgcn_swmmac_f32_16x16x32_bf8_fp8_w64(ai, b2i, c4f, index); // expected-error{{'__builtin_amdgcn_swmmac_f32_16x16x32_bf8_fp8_w64' needs target feature swmmac-gfx1200-insts,wavefrontsize64}}
  *out4f = __builtin_amdgcn_swmmac_f32_16x16x32_bf8_bf8_w64(ai, b2i, c4f, index); // expected-error{{'__builtin_amdgcn_swmmac_f32_16x16x32_bf8_bf8_w64' needs target feature swmmac-gfx1200-insts,wavefrontsize64}}
}
