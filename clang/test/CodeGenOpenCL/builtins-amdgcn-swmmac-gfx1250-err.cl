// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1030 \
// RUN:   -verify -S -o - %s

typedef float  v8f    __attribute__((ext_vector_type(8)));
typedef half   v8h    __attribute__((ext_vector_type(8)));
typedef half   v16h   __attribute__((ext_vector_type(16)));
typedef half   v32h   __attribute__((ext_vector_type(32)));
typedef __bf16 v8bf16  __attribute__((ext_vector_type(8)));
typedef __bf16 v16bf16 __attribute__((ext_vector_type(16)));
typedef __bf16 v32bf16 __attribute__((ext_vector_type(32)));
typedef int    v2i    __attribute__((ext_vector_type(2)));
typedef int    v8i    __attribute__((ext_vector_type(8)));
typedef int    v16i   __attribute__((ext_vector_type(16)));

void test_amdgcn_swmmac_gfx1250(global v8f* out8f, global v8h* out8h, global v8bf16* out8bf16, global v8i* out8i,
                                  v16bf16 a16bf16, v16h a16h, v8i a8i,
                                  v32bf16 b32bf16, v32h b32h, v16i b16i,
                                  v8f c8f, v8bf16 c8bf16, v8h c8h, v8i c8i,
                                  int index, v2i index2)
{
  *out8f = __builtin_amdgcn_swmmac_f32_16x16x64_bf16(0, a16bf16, 0, b32bf16, c8f, index, false, true); // expected-error{{'__builtin_amdgcn_swmmac_f32_16x16x64_bf16' needs target feature swmmac-gfx1250-insts,wavefrontsize32}}
  *out8bf16 = __builtin_amdgcn_swmmac_bf16_16x16x64_bf16(0, a16bf16, 0, b32bf16, c8bf16, index, false, true); // expected-error{{'__builtin_amdgcn_swmmac_bf16_16x16x64_bf16' needs target feature swmmac-gfx1250-insts,wavefrontsize32}}
  *out8f = __builtin_amdgcn_swmmac_bf16f32_16x16x64_bf16(0, a16bf16, 0, b32bf16, c8f, index, false, true); // expected-error{{'__builtin_amdgcn_swmmac_bf16f32_16x16x64_bf16' needs target feature swmmac-gfx1250-insts,wavefrontsize32}}
  *out8f = __builtin_amdgcn_swmmac_f32_16x16x128_fp8_fp8(a8i, b16i, c8f, index2, false, true); // expected-error{{'__builtin_amdgcn_swmmac_f32_16x16x128_fp8_fp8' needs target feature swmmac-gfx1250-insts,wavefrontsize32}}
  *out8f = __builtin_amdgcn_swmmac_f32_16x16x128_fp8_bf8(a8i, b16i, c8f, index2, false, true); // expected-error{{'__builtin_amdgcn_swmmac_f32_16x16x128_fp8_bf8' needs target feature swmmac-gfx1250-insts,wavefrontsize32}}
  *out8f = __builtin_amdgcn_swmmac_f32_16x16x128_bf8_fp8(a8i, b16i, c8f, index2, false, true); // expected-error{{'__builtin_amdgcn_swmmac_f32_16x16x128_bf8_fp8' needs target feature swmmac-gfx1250-insts,wavefrontsize32}}
  *out8f = __builtin_amdgcn_swmmac_f32_16x16x128_bf8_bf8(a8i, b16i, c8f, index2, false, true); // expected-error{{'__builtin_amdgcn_swmmac_f32_16x16x128_bf8_bf8' needs target feature swmmac-gfx1250-insts,wavefrontsize32}}
  *out8h = __builtin_amdgcn_swmmac_f16_16x16x128_fp8_fp8(a8i, b16i, c8h, index2, false, true); // expected-error{{'__builtin_amdgcn_swmmac_f16_16x16x128_fp8_fp8' needs target feature swmmac-gfx1250-insts,wavefrontsize32}}
  *out8h = __builtin_amdgcn_swmmac_f16_16x16x128_fp8_bf8(a8i, b16i, c8h, index2, false, true); // expected-error{{'__builtin_amdgcn_swmmac_f16_16x16x128_fp8_bf8' needs target feature swmmac-gfx1250-insts,wavefrontsize32}}
  *out8h = __builtin_amdgcn_swmmac_f16_16x16x128_bf8_fp8(a8i, b16i, c8h, index2, false, true); // expected-error{{'__builtin_amdgcn_swmmac_f16_16x16x128_bf8_fp8' needs target feature swmmac-gfx1250-insts,wavefrontsize32}}
  *out8h = __builtin_amdgcn_swmmac_f16_16x16x128_bf8_bf8(a8i, b16i, c8h, index2, false, true); // expected-error{{'__builtin_amdgcn_swmmac_f16_16x16x128_bf8_bf8' needs target feature swmmac-gfx1250-insts,wavefrontsize32}}
  *out8i = __builtin_amdgcn_swmmac_i32_16x16x128_iu8(0, a8i, 0, b16i, c8i, index2, false, true); // expected-error{{'__builtin_amdgcn_swmmac_i32_16x16x128_iu8' needs target feature swmmac-gfx1250-insts,wavefrontsize32}}
  *out8f = __builtin_amdgcn_swmmac_f32_16x16x64_f16(0, a16h, 0, b32h, c8f, index, false, true); // expected-error{{'__builtin_amdgcn_swmmac_f32_16x16x64_f16' needs target feature swmmac-gfx1250-insts,wavefrontsize32}}
  *out8h = __builtin_amdgcn_swmmac_f16_16x16x64_f16(0, a16h, 0, b32h, c8h, index, false, true); // expected-error{{'__builtin_amdgcn_swmmac_f16_16x16x64_f16' needs target feature swmmac-gfx1250-insts,wavefrontsize32}}
}
