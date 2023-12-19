// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-- -target-cpu gfx940 -verify -S -o - %s

typedef float float4 __attribute__((ext_vector_type(4)));
typedef float float16 __attribute__((ext_vector_type(16)));
typedef half half8 __attribute__((ext_vector_type(8)));
typedef __bf16 bfloat8 __attribute__((ext_vector_type(8)));

void test(__global float4* out0, half8 a0, half8 b0, float4 c0,
          __global float16* out1, half8 a1, half8 b1, float16 c1,
          __global float16* out2, bfloat8 a2, bfloat8 b2, float16 c2) {
  *out0 = __builtin_amdgcn_mfma_f32_16x16x32_f16(a0, b0, c0, 0, 0, 0); // expected-error{{'__builtin_amdgcn_mfma_f32_16x16x32_f16' needs target feature gfx950-insts}}
  *out1 = __builtin_amdgcn_mfma_f32_32x32x16_f16(a1, b1, c1, 0, 0, 0); // expected-error{{'__builtin_amdgcn_mfma_f32_32x32x16_f16' needs target feature gfx950-insts}}
  *out2 = __builtin_amdgcn_mfma_f32_32x32x16_bf16(a2, b2, c2, 0, 0, 0); // expected-error{{'__builtin_amdgcn_mfma_f32_32x32x16_bf16' needs target feature gfx950-insts}}
}
