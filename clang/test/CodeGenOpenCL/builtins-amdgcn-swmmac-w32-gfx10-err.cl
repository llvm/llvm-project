// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1030 \
// RUN:   -verify -S -o - %s

typedef int    v2i   __attribute__((ext_vector_type(2)));
typedef int    v4i   __attribute__((ext_vector_type(4)));
typedef float  v8f   __attribute__((ext_vector_type(8)));
typedef half   v8h   __attribute__((ext_vector_type(8)));
typedef short  v8s   __attribute__((ext_vector_type(8)));
typedef int    v8i   __attribute__((ext_vector_type(8)));
typedef half  v16h   __attribute__((ext_vector_type(16)));
typedef short v16s   __attribute__((ext_vector_type(16)));

void test_amdgcn_swmmac_w32(global v8f* out8f, global v8h* out8h, global v8s* out8s, global v8i* out8i,
                             v8h a8h, v8s a8s, v2i a2i, int ai,
                             v16h b16h, v16s b16s, v4i b4i, v2i b2i,
                             v8f c8f, v8h c8h, v8s c8s, v8i c8i,
                             int index)
{
  *out8f = __builtin_amdgcn_swmmac_f32_16x16x32_f16_w32(a8h, b16h, c8f, index); // expected-error{{'__builtin_amdgcn_swmmac_f32_16x16x32_f16_w32' needs target feature swmmac-gfx1200-insts,wavefrontsize32}}
  *out8f = __builtin_amdgcn_swmmac_f32_16x16x32_bf16_w32(a8s, b16s, c8f, index); // expected-error{{'__builtin_amdgcn_swmmac_f32_16x16x32_bf16_w32' needs target feature swmmac-gfx1200-insts,wavefrontsize32}}
  *out8h = __builtin_amdgcn_swmmac_f16_16x16x32_f16_w32(a8h, b16h, c8h, index); // expected-error{{'__builtin_amdgcn_swmmac_f16_16x16x32_f16_w32' needs target feature swmmac-gfx1200-insts,wavefrontsize32}}
  *out8s = __builtin_amdgcn_swmmac_bf16_16x16x32_bf16_w32(a8s, b16s, c8s, index); // expected-error{{'__builtin_amdgcn_swmmac_bf16_16x16x32_bf16_w32' needs target feature swmmac-gfx1200-insts,wavefrontsize32}}
  *out8i = __builtin_amdgcn_swmmac_i32_16x16x32_iu8_w32(true, a2i, true, b4i, c8i, index, true); // expected-error{{'__builtin_amdgcn_swmmac_i32_16x16x32_iu8_w32' needs target feature swmmac-gfx1200-insts,wavefrontsize32}}
  *out8i = __builtin_amdgcn_swmmac_i32_16x16x32_iu4_w32(true, ai, true, b2i, c8i, index, true); // expected-error{{'__builtin_amdgcn_swmmac_i32_16x16x32_iu4_w32' needs target feature swmmac-gfx1200-insts,wavefrontsize32}}
  *out8i = __builtin_amdgcn_swmmac_i32_16x16x64_iu4_w32(true, a2i, true, b4i, c8i, index, true); // expected-error{{'__builtin_amdgcn_swmmac_i32_16x16x64_iu4_w32' needs target feature swmmac-gfx1200-insts,wavefrontsize32}}
  *out8f = __builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32(a2i, b4i, c8f, index); // expected-error{{'__builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32' needs target feature swmmac-gfx1200-insts,wavefrontsize32}}
  *out8f = __builtin_amdgcn_swmmac_f32_16x16x32_fp8_bf8_w32(a2i, b4i, c8f, index); // expected-error{{'__builtin_amdgcn_swmmac_f32_16x16x32_fp8_bf8_w32' needs target feature swmmac-gfx1200-insts,wavefrontsize32}}
  *out8f = __builtin_amdgcn_swmmac_f32_16x16x32_bf8_fp8_w32(a2i, b4i, c8f, index); // expected-error{{'__builtin_amdgcn_swmmac_f32_16x16x32_bf8_fp8_w32' needs target feature swmmac-gfx1200-insts,wavefrontsize32}}
  *out8f = __builtin_amdgcn_swmmac_f32_16x16x32_bf8_bf8_w32(a2i, b4i, c8f, index); // expected-error{{'__builtin_amdgcn_swmmac_f32_16x16x32_bf8_bf8_w32' needs target feature swmmac-gfx1200-insts,wavefrontsize32}}
}
