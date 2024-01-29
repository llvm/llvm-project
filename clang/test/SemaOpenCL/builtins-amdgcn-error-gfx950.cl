// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-- -target-cpu gfx940 -verify -S -o - %s

typedef float float4 __attribute__((ext_vector_type(4)));
typedef float float16 __attribute__((ext_vector_type(16)));
typedef half half8 __attribute__((ext_vector_type(8)));
typedef __bf16 bfloat8 __attribute__((ext_vector_type(8)));
typedef int int4 __attribute__((ext_vector_type(4)));
typedef int int16 __attribute__((ext_vector_type(16)));

void test(__global float4* out0, half8 a0, half8 b0, float4 c0,
          __global float16* out1, half8 a1, half8 b1, float16 c1,
          __global float16* out2, bfloat8 a2, bfloat8 b2, float16 c2,
          __global int4* out3, int4 a3, int4 b3, int4 c3,
          __global int16* out4, int4 a4, int4 b4, int16 c4) {
  *out0 = __builtin_amdgcn_mfma_f32_16x16x32_f16(a0, b0, c0, 0, 0, 0); // expected-error{{'__builtin_amdgcn_mfma_f32_16x16x32_f16' needs target feature gfx950-insts}}
  *out1 = __builtin_amdgcn_mfma_f32_32x32x16_f16(a1, b1, c1, 0, 0, 0); // expected-error{{'__builtin_amdgcn_mfma_f32_32x32x16_f16' needs target feature gfx950-insts}}
  *out2 = __builtin_amdgcn_mfma_f32_32x32x16_bf16(a2, b2, c2, 0, 0, 0); // expected-error{{'__builtin_amdgcn_mfma_f32_32x32x16_bf16' needs target feature gfx950-insts}}
  *out3 = __builtin_amdgcn_mfma_i32_16x16x64_i8(a3, b3, c3, 0, 0, 0); // expected-error{{'__builtin_amdgcn_mfma_i32_16x16x64_i8' needs target feature gfx950-insts}}
  *out4 = __builtin_amdgcn_mfma_i32_32x32x32_i8(a4, b4, c4, 0, 0, 0); // expected-error{{'__builtin_amdgcn_mfma_i32_32x32x32_i8' needs target feature gfx950-insts}}
}
