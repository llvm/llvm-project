// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx9-4-generic -verify -emit-llvm -o - %s

typedef unsigned int uint;
typedef float  float2   __attribute__((ext_vector_type(2)));
typedef float  float4   __attribute__((ext_vector_type(4)));
typedef float  float16  __attribute__((ext_vector_type(16)));
typedef int    int2   __attribute__((ext_vector_type(2)));
typedef int    int4   __attribute__((ext_vector_type(4)));

void builtin_test_unsupported(uint a, uint b, int a_int, long  a_long, float a_float, float b_float,
                              int2 a_int2, int4 a_int4, float2 a_float2, float4 a_float4, float16 a_float16) {
  a_float4 = __builtin_amdgcn_mfma_f32_16x16x32_bf8_bf8(a_long, a_long, a_float4, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_16x16x32_bf8_bf8' needs target feature fp8-insts}}
  a_float4 = __builtin_amdgcn_mfma_f32_16x16x32_bf8_fp8(a_long, a_long, a_float4, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_16x16x32_bf8_fp8' needs target feature fp8-insts}}
  a_float4 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_bf8(a_long, a_long, a_float4, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_16x16x32_fp8_bf8' needs target feature fp8-insts}}
  a_float4 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a_long, a_long, a_float4, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8' needs target feature fp8-insts}}
  a_float16 = __builtin_amdgcn_mfma_f32_32x32x16_bf8_bf8(a_long, a_long, a_float16, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_32x32x16_bf8_bf8' needs target feature fp8-insts}}
  a_float16 = __builtin_amdgcn_mfma_f32_32x32x16_bf8_fp8(a_long, a_long, a_float16, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_32x32x16_bf8_fp8' needs target feature fp8-insts}}
  a_float16 = __builtin_amdgcn_mfma_f32_32x32x16_fp8_bf8(a_long, a_long, a_float16, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_32x32x16_fp8_bf8' needs target feature fp8-insts}}
  a_float16 = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(a_long, a_long, a_float16, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8' needs target feature fp8-insts}}
  a_float4 = __builtin_amdgcn_smfmac_f32_16x16x64_bf8_bf8(a_int2, a_int4, a_float4, a_int, 0, 0); // expected-error {{'__builtin_amdgcn_smfmac_f32_16x16x64_bf8_bf8' needs target feature fp8-insts}}
  a_float4 = __builtin_amdgcn_smfmac_f32_16x16x64_bf8_fp8(a_int2, a_int4, a_float4, a_int, 0, 0); // expected-error {{'__builtin_amdgcn_smfmac_f32_16x16x64_bf8_fp8' needs target feature fp8-insts}}
  a_float4 = __builtin_amdgcn_smfmac_f32_16x16x64_fp8_bf8(a_int2, a_int4, a_float4, a_int, 0, 0); // expected-error {{'__builtin_amdgcn_smfmac_f32_16x16x64_fp8_bf8' needs target feature fp8-insts}}
  a_float4 = __builtin_amdgcn_smfmac_f32_16x16x64_fp8_fp8(a_int2, a_int4, a_float4, a_int, 0, 0); // expected-error {{'__builtin_amdgcn_smfmac_f32_16x16x64_fp8_fp8' needs target feature fp8-insts}}
  a_float16 = __builtin_amdgcn_smfmac_f32_32x32x32_bf8_bf8(a_int2, a_int4, a_float16, a_int, 0, 0); // expected-error {{'__builtin_amdgcn_smfmac_f32_32x32x32_bf8_bf8' needs target feature fp8-insts}}
  a_float16 = __builtin_amdgcn_smfmac_f32_32x32x32_bf8_fp8(a_int2, a_int4, a_float16, a_int, 0, 0); // expected-error {{'__builtin_amdgcn_smfmac_f32_32x32x32_bf8_fp8' needs target feature fp8-insts}}
  a_float16 = __builtin_amdgcn_smfmac_f32_32x32x32_fp8_bf8(a_int2, a_int4, a_float16, a_int, 0, 0); // expected-error {{'__builtin_amdgcn_smfmac_f32_32x32x32_fp8_bf8' needs target feature fp8-insts}}
  a_float16 = __builtin_amdgcn_smfmac_f32_32x32x32_fp8_fp8(a_int2, a_int4, a_float16, a_int, 0, 0); // expected-error {{'__builtin_amdgcn_smfmac_f32_32x32x32_fp8_fp8' needs target feature fp8-insts}}
  b = __builtin_amdgcn_cvt_f32_bf8(a, 0); // expected-error {{'__builtin_amdgcn_cvt_f32_bf8' needs target feature fp8-conversion-insts}}
  b = __builtin_amdgcn_cvt_f32_fp8(a, 1); // expected-error {{'__builtin_amdgcn_cvt_f32_fp8' needs target feature fp8-conversion-insts}}
  a_float2 = __builtin_amdgcn_cvt_pk_f32_bf8(a, false); // expected-error {{'__builtin_amdgcn_cvt_pk_f32_bf8' needs target feature fp8-conversion-insts}}
  a_float2 = __builtin_amdgcn_cvt_pk_f32_fp8(a, true); // expected-error {{'__builtin_amdgcn_cvt_pk_f32_fp8' needs target feature fp8-conversion-insts}}
  b = __builtin_amdgcn_cvt_pk_bf8_f32(a_float, b_float, a, false); // expected-error {{'__builtin_amdgcn_cvt_pk_bf8_f32' needs target feature fp8-conversion-insts}}
  b = __builtin_amdgcn_cvt_pk_fp8_f32(a_float, b_float, a, true); // expected-error {{'__builtin_amdgcn_cvt_pk_fp8_f32' needs target feature fp8-conversion-insts}}
  b = __builtin_amdgcn_cvt_sr_bf8_f32(a_float, b_float, a, 2); // expected-error {{'__builtin_amdgcn_cvt_sr_bf8_f32' needs target feature fp8-conversion-insts}}
  b = __builtin_amdgcn_cvt_sr_fp8_f32(a_float, b_float, a, 3); // expected-error {{'__builtin_amdgcn_cvt_sr_fp8_f32' needs target feature fp8-conversion-insts}}
}
