// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-- -target-cpu gfx942 -verify -S -o - %s

typedef float float4 __attribute__((ext_vector_type(4)));
typedef float float16 __attribute__((ext_vector_type(16)));
typedef half half8 __attribute__((ext_vector_type(8)));
typedef half half16 __attribute__((ext_vector_type(16)));
typedef __bf16 bfloat8 __attribute__((ext_vector_type(8)));
typedef __bf16 bfloat16 __attribute__((ext_vector_type(16)));
typedef unsigned int uint2 __attribute__((ext_vector_type(2)));
typedef int int4 __attribute__((ext_vector_type(4)));
typedef int int8 __attribute__((ext_vector_type(8)));
typedef int int16 __attribute__((ext_vector_type(16)));

void test(__global float4* out0, half8 a0, half8 b0, float4 c0,
          __global float16* out1, half8 a1, half8 b1, float16 c1,
          __global float16* out2, bfloat8 a2, bfloat8 b2, float16 c2,
          __global int4* out3, int4 a3, int4 b3, int4 c3,
          __global int16* out4, int4 a4, int4 b4, int16 c4,
          __global float4* out5, bfloat8 a5, bfloat8 b5, float4 c5,
          __global float4* out6, half8 a6, half16 b6, float4 c6,
          __global float16* out7, half8 a7, half16 b7, float16 c7,
          __global float4* out8, bfloat8 a8, bfloat16 b8, float4 c8,
          __global float16* out9, bfloat8 a9, bfloat16 b9, float16 c9,
          __global int4* out10, int4 a10, int8 b10, int4 c10,
          __global int16* out11, int4 a11, int8 b11, int16 c11,
          __global float4* out12, int4 a12, int8 b12, float4 c12,
          __global float16* out13, int4 a13, int8 b13, float16 c13,
          __global float4* out14, int8 a14, int8 b14, float4 c14, int d14, int e14,
          __global float16* out15, int8 a15, int8 b15, float16 c15, int d15, int e15,
          __global uint2* out16, int a16, int b16,
          __global int *out17, float a17, int b17, float c17) {
  *out0 = __builtin_amdgcn_mfma_f32_16x16x32_f16(a0, b0, c0, 0, 0, 0); // expected-error{{'__builtin_amdgcn_mfma_f32_16x16x32_f16' needs target feature gfx950-insts}}
  *out1 = __builtin_amdgcn_mfma_f32_32x32x16_f16(a1, b1, c1, 0, 0, 0); // expected-error{{'__builtin_amdgcn_mfma_f32_32x32x16_f16' needs target feature gfx950-insts}}
  *out2 = __builtin_amdgcn_mfma_f32_32x32x16_bf16(a2, b2, c2, 0, 0, 0); // expected-error{{'__builtin_amdgcn_mfma_f32_32x32x16_bf16' needs target feature gfx950-insts}}
  *out3 = __builtin_amdgcn_mfma_i32_16x16x64_i8(a3, b3, c3, 0, 0, 0); // expected-error{{'__builtin_amdgcn_mfma_i32_16x16x64_i8' needs target feature gfx950-insts}}
  *out4 = __builtin_amdgcn_mfma_i32_32x32x32_i8(a4, b4, c4, 0, 0, 0); // expected-error{{'__builtin_amdgcn_mfma_i32_32x32x32_i8' needs target feature gfx950-insts}}
  *out5 = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a5, b5, c5, 0, 0, 0); // expected-error{{'__builtin_amdgcn_mfma_f32_16x16x32_bf16' needs target feature gfx950-insts}}
  *out6 = __builtin_amdgcn_smfmac_f32_16x16x64_f16(a6, b6, c6, 0, 0, 0); // expected-error{{'__builtin_amdgcn_smfmac_f32_16x16x64_f16' needs target feature gfx950-insts}}
  *out7 = __builtin_amdgcn_smfmac_f32_32x32x32_f16(a7, b7, c7, 0, 0, 0); // expected-error{{'__builtin_amdgcn_smfmac_f32_32x32x32_f16' needs target feature gfx950-insts}}
  *out8 = __builtin_amdgcn_smfmac_f32_16x16x64_bf16(a8, b8, c8, 0, 0, 0); // expected-error{{'__builtin_amdgcn_smfmac_f32_16x16x64_bf16' needs target feature gfx950-insts}}
  *out9 = __builtin_amdgcn_smfmac_f32_32x32x32_bf16(a9, b9, c9, 0, 0, 0); // expected-error{{'__builtin_amdgcn_smfmac_f32_32x32x32_bf16' needs target feature gfx950-insts}}
  *out10 = __builtin_amdgcn_smfmac_i32_16x16x128_i8(a10, b10, c10, 0, 0, 0); // expected-error{{'__builtin_amdgcn_smfmac_i32_16x16x128_i8' needs target feature gfx950-insts}}
  *out11 = __builtin_amdgcn_smfmac_i32_32x32x64_i8(a11, b11, c11, 0, 0, 0); // expected-error{{'__builtin_amdgcn_smfmac_i32_32x32x64_i8' needs target feature gfx950-insts}}
  *out12 = __builtin_amdgcn_smfmac_f32_16x16x128_bf8_bf8(a12, b12, c12, 0, 0, 0); // expected-error{{'__builtin_amdgcn_smfmac_f32_16x16x128_bf8_bf8' needs target feature gfx950-insts}}
  *out12 = __builtin_amdgcn_smfmac_f32_16x16x128_bf8_fp8(a12, b12, c12, 0, 0, 0); // expected-error{{'__builtin_amdgcn_smfmac_f32_16x16x128_bf8_fp8' needs target feature gfx950-insts}}
  *out12 = __builtin_amdgcn_smfmac_f32_16x16x128_fp8_bf8(a12, b12, c12, 0, 0, 0); // expected-error{{'__builtin_amdgcn_smfmac_f32_16x16x128_fp8_bf8' needs target feature gfx950-insts}}
  *out12 = __builtin_amdgcn_smfmac_f32_16x16x128_fp8_fp8(a12, b12, c12, 0, 0, 0); // expected-error{{'__builtin_amdgcn_smfmac_f32_16x16x128_fp8_fp8' needs target feature gfx950-insts}}
  *out13 = __builtin_amdgcn_smfmac_f32_32x32x64_bf8_bf8(a13, b13, c13, 0, 0, 0); // expected-error{{'__builtin_amdgcn_smfmac_f32_32x32x64_bf8_bf8' needs target feature gfx950-insts}}
  *out13 = __builtin_amdgcn_smfmac_f32_32x32x64_bf8_fp8(a13, b13, c13, 0, 0, 0); // expected-error{{'__builtin_amdgcn_smfmac_f32_32x32x64_bf8_fp8' needs target feature gfx950-insts}}
  *out13 = __builtin_amdgcn_smfmac_f32_32x32x64_fp8_bf8(a13, b13, c13, 0, 0, 0); // expected-error{{'__builtin_amdgcn_smfmac_f32_32x32x64_fp8_bf8' needs target feature gfx950-insts}}
  *out13 = __builtin_amdgcn_smfmac_f32_32x32x64_fp8_fp8(a13, b13, c13, 0, 0, 0); // expected-error{{'__builtin_amdgcn_smfmac_f32_32x32x64_fp8_fp8' needs target feature gfx950-insts}}
  *out14 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a14, b14, c14, 0, 0, 0, d14, 0, e14); // expected-error{{'__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4' needs target feature gfx950-insts}}
  *out15 = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(a15, b15, c15, 0, 0, 0, d15, 0, e15); // expected-error{{'__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4' needs target feature gfx950-insts}}
  *out16 = __builtin_amdgcn_permlane16_swap(a16, b16, false, false); // expected-error{{'__builtin_amdgcn_permlane16_swap' needs target feature permlane16-swap}}
  *out16 = __builtin_amdgcn_permlane32_swap(a16, b16, false, false); // expected-error{{'__builtin_amdgcn_permlane32_swap' needs target feature permlane32-swap}}
  *out17 = __builtin_amdgcn_cvt_scalef32_sr_bf8_bf16(*out17, a17, b17, c17, 0); // expected-error{{'__builtin_amdgcn_cvt_scalef32_sr_bf8_bf16' needs target feature bf8-cvt-scale-insts}}
  *out17 = __builtin_amdgcn_cvt_scalef32_sr_bf8_f16(*out17, a17, b17, c17, 0); // expected-error{{'__builtin_amdgcn_cvt_scalef32_sr_bf8_f16' needs target feature bf8-cvt-scale-insts}}
  *out17 = __builtin_amdgcn_cvt_scalef32_sr_bf8_f32(*out17, a17, b17, c17, 0); // expected-error{{'__builtin_amdgcn_cvt_scalef32_sr_bf8_f32' needs target feature bf8-cvt-scale-insts}}
  *out17 = __builtin_amdgcn_cvt_scalef32_sr_fp8_bf16(*out17, a17, b17, c17, 0); // expected-error{{'__builtin_amdgcn_cvt_scalef32_sr_fp8_bf16' needs target feature fp8-cvt-scale-insts}}
  *out17 = __builtin_amdgcn_cvt_scalef32_sr_fp8_f16(*out17, a17, b17, c17, 0); // expected-error{{'__builtin_amdgcn_cvt_scalef32_sr_fp8_f16' needs target feature fp8-cvt-scale-insts}}
  *out17 = __builtin_amdgcn_cvt_scalef32_sr_fp8_f32(*out17, a17, b17, c17, 0); // expected-error{{'__builtin_amdgcn_cvt_scalef32_sr_fp8_f32' needs target feature fp8-cvt-scale-insts}}
}
