// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-- -target-cpu gfx950 -verify -S -o - %s

typedef float float4 __attribute__((ext_vector_type(4)));
typedef float float16 __attribute__((ext_vector_type(16)));
typedef half half8 __attribute__((ext_vector_type(8)));
typedef half half16 __attribute__((ext_vector_type(16)));
typedef __bf16 bfloat8 __attribute__((ext_vector_type(8)));
typedef __bf16 bfloat16 __attribute__((ext_vector_type(16)));
typedef int int4 __attribute__((ext_vector_type(4)));
typedef int int8 __attribute__((ext_vector_type(8)));
typedef int int16 __attribute__((ext_vector_type(16)));
typedef unsigned int uint;
typedef half half2 __attribute__((ext_vector_type(2)));
typedef short short2 __attribute__((ext_vector_type(2)));
typedef float float2 __attribute__((ext_vector_type(2)));
typedef __bf16 bfloat2 __attribute__((ext_vector_type(2)));
typedef unsigned short ushort;

void test_mfma_f32_16x16x32_f16(__global float4* out, half8 a, half8 b, float4 c, int X) {

  *out = __builtin_amdgcn_mfma_f32_16x16x32_f16(a, b, c, X, 0, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_16x16x32_f16' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_16x16x32_f16(a, b, c, 0, X, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_16x16x32_f16' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_16x16x32_f16(a, b, c, 0, 0, X); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_16x16x32_f16' must be a constant integer}}
}


void test_mfma_f32_32x32x16_f16(__global float16* out, half8 a, half8 b, float16 c, int X) {
  *out = __builtin_amdgcn_mfma_f32_32x32x16_f16(a, b, c, X, 0, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_32x32x16_f16' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_32x32x16_f16(a, b, c, 0, X, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_32x32x16_f16' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_32x32x16_f16(a, b, c, 0, 0, X); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_32x32x16_f16' must be a constant integer}}
}

void test_mfma_f32_32x32x16_bf16(__global float16* out, bfloat8 a, bfloat8 b, float16 c, int X) {
  *out = __builtin_amdgcn_mfma_f32_32x32x16_bf16(a, b, c, X, 0, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_32x32x16_bf16' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_32x32x16_bf16(a, b, c, 0, X, 0);  // expected-error{{argument to '__builtin_amdgcn_mfma_f32_32x32x16_bf16' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_32x32x16_bf16(a, b, c, 0, 0, X);  // expected-error{{argument to '__builtin_amdgcn_mfma_f32_32x32x16_bf16' must be a constant integer}}
}

void test_mfma_scale_f32_16x16x128_f8f6f4(__global float4* out, int8 a, int8 b, float4 c, int X, int Y) {
  *out = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b, c, X, 0, 1, Y, 2, Y); // expected-error{{argument to '__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b, c, 0, X, 1, Y, 2, Y); // expected-error{{argument to '__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b, c, 0, 0, X, Y, 2, Y); // expected-error{{argument to '__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b, c, 0, 0, 0, Y, X, Y); // expected-error{{argument to '__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4' must be a constant integer}}
}

void test_mfma_scale_f32_32x32x64_f8f6f4(__global float16* out, int8 a, int8 b, float16 c, int X, int Y) {
  *out = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(a, b, c, X, 0, 1, Y, 2, Y); // expected-error{{argument to '__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(a, b, c, 0, X, 1, Y, 2, Y); // expected-error{{argument to '__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(a, b, c, 0, 0, X, Y, 2, Y); // expected-error{{argument to '__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(a, b, c, 0, 0, 0, Y, X, Y); // expected-error{{argument to '__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4' must be a constant integer}}
}

void test_mfma_i32_16x16x64_i8(__global int4* out, int4 a, int4 b, int4 c, int X) {
  *out = __builtin_amdgcn_mfma_i32_16x16x64_i8(a, b, c, X, 0, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_i32_16x16x64_i8' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_i32_16x16x64_i8(a, b, c, 0, X, 0);  // expected-error{{argument to '__builtin_amdgcn_mfma_i32_16x16x64_i8' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_i32_16x16x64_i8(a, b, c, 0, 0, X);  // expected-error{{argument to '__builtin_amdgcn_mfma_i32_16x16x64_i8' must be a constant integer}}
}

void test_mfma_i32_32x32x32_i8(__global int16* out, int4 a, int4 b, int16 c, int X) {
  *out = __builtin_amdgcn_mfma_i32_32x32x32_i8(a, b, c, X, 0, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_i32_32x32x32_i8' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_i32_32x32x32_i8(a, b, c, 0, X, 0);  // expected-error{{argument to '__builtin_amdgcn_mfma_i32_32x32x32_i8' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_i32_32x32x32_i8(a, b, c, 0, 0, X);  // expected-error{{argument to '__builtin_amdgcn_mfma_i32_32x32x32_i8' must be a constant integer}}
}

void test_mfma_f32_16x16x32_bf16(__global float4* out, bfloat8 a, bfloat8 b, float4 c, int X) {

  *out = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, c, X, 0, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_16x16x32_bf16' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, c, 0, X, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_16x16x32_bf16' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, c, 0, 0, X); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_16x16x32_bf16' must be a constant integer}}
}

void test_smfmac_f32_16x16x64_f16(global float4* out, half8 a, half16 b, float4 c, int idx, int d)
{
  *out = __builtin_amdgcn_smfmac_f32_16x16x64_f16(a, b, c, idx, d, 0); // expected-error{{argument to '__builtin_amdgcn_smfmac_f32_16x16x64_f16' must be a constant integer}}
  *out = __builtin_amdgcn_smfmac_f32_16x16x64_f16(a, b, c, idx, 0, d); // expected-error{{argument to '__builtin_amdgcn_smfmac_f32_16x16x64_f16' must be a constant integer}}
}

void test_smfmac_f32_32x32x32_f16(global float16* out, half8 a, half16 b, float16 c, int idx, int d)
{
  *out = __builtin_amdgcn_smfmac_f32_32x32x32_f16(a, b, c, idx, d, 0); // expected-error{{argument to '__builtin_amdgcn_smfmac_f32_32x32x32_f16' must be a constant integer}}
  *out = __builtin_amdgcn_smfmac_f32_32x32x32_f16(a, b, c, idx, 0, d); // expected-error{{argument to '__builtin_amdgcn_smfmac_f32_32x32x32_f16' must be a constant integer}}
}

void test_smfmac_f32_16x16x64_bf16(global float4* out, bfloat8 a, bfloat16 b, float4 c, int idx, int d)
{
  *out = __builtin_amdgcn_smfmac_f32_16x16x64_bf16(a, b, c, idx, d, 0); // expected-error{{argument to '__builtin_amdgcn_smfmac_f32_16x16x64_bf16' must be a constant integer}}
  *out = __builtin_amdgcn_smfmac_f32_16x16x64_bf16(a, b, c, idx, 0, d); // expected-error{{argument to '__builtin_amdgcn_smfmac_f32_16x16x64_bf16' must be a constant integer}}
}

void test_smfmac_f32_32x32x32_bf16(global float16* out, bfloat8 a, bfloat16 b, float16 c, int idx, int d)
{
  *out = __builtin_amdgcn_smfmac_f32_32x32x32_bf16(a, b, c, idx, d, 0); // expected-error{{argument to '__builtin_amdgcn_smfmac_f32_32x32x32_bf16' must be a constant integer}}
  *out = __builtin_amdgcn_smfmac_f32_32x32x32_bf16(a, b, c, idx, 0, d); // expected-error{{argument to '__builtin_amdgcn_smfmac_f32_32x32x32_bf16' must be a constant integer}}
}

void test_smfmac_i32_16x16x128_i8(global int4* out, int4 a, int8 b, int4 c, int idx, int d)
{
  *out = __builtin_amdgcn_smfmac_i32_16x16x128_i8(a, b, c, idx, d, 0); // expected-error{{argument to '__builtin_amdgcn_smfmac_i32_16x16x128_i8' must be a constant integer}}
  *out = __builtin_amdgcn_smfmac_i32_16x16x128_i8(a, b, c, idx, 0, d); // expected-error{{argument to '__builtin_amdgcn_smfmac_i32_16x16x128_i8' must be a constant integer}}
}

void test_smfmac_i32_32x32x64_i8(global int16* out, int4 a, int8 b, int16 c, int idx, int d)
{
  *out = __builtin_amdgcn_smfmac_i32_32x32x64_i8(a, b, c, idx, d, 0); // expected-error{{argument to '__builtin_amdgcn_smfmac_i32_32x32x64_i8' must be a constant integer}}
  *out = __builtin_amdgcn_smfmac_i32_32x32x64_i8(a, b, c, idx, 0, d); // expected-error{{argument to '__builtin_amdgcn_smfmac_i32_32x32x64_i8' must be a constant integer}}
}

void test_smfmac_f32_16x16x128_bf8_bf8(global float4* out, int4 a, int8 b, float4 c, int idx, int d)
{
  *out = __builtin_amdgcn_smfmac_f32_16x16x128_bf8_bf8(a, b, c, idx, d, 0); // expected-error{{argument to '__builtin_amdgcn_smfmac_f32_16x16x128_bf8_bf8' must be a constant integer}}
  *out = __builtin_amdgcn_smfmac_f32_16x16x128_bf8_bf8(a, b, c, idx, 0, d); // expected-error{{argument to '__builtin_amdgcn_smfmac_f32_16x16x128_bf8_bf8' must be a constant integer}}
}

void test_smfmac_f32_16x16x128_bf8_fp8(global float4* out, int4 a, int8 b, float4 c, int idx, int d)
{
  *out = __builtin_amdgcn_smfmac_f32_16x16x128_bf8_fp8(a, b, c, idx, d, 0); // expected-error{{argument to '__builtin_amdgcn_smfmac_f32_16x16x128_bf8_fp8' must be a constant integer}}
  *out = __builtin_amdgcn_smfmac_f32_16x16x128_bf8_fp8(a, b, c, idx, 0, d); // expected-error{{argument to '__builtin_amdgcn_smfmac_f32_16x16x128_bf8_fp8' must be a constant integer}}
}

void test_smfmac_f32_16x16x128_fp8_bf8(global float4* out, int4 a, int8 b, float4 c, int idx, int d)
{
  *out = __builtin_amdgcn_smfmac_f32_16x16x128_fp8_bf8(a, b, c, idx, d, 0); // expected-error{{argument to '__builtin_amdgcn_smfmac_f32_16x16x128_fp8_bf8' must be a constant integer}}
  *out = __builtin_amdgcn_smfmac_f32_16x16x128_fp8_bf8(a, b, c, idx, 0, d); // expected-error{{argument to '__builtin_amdgcn_smfmac_f32_16x16x128_fp8_bf8' must be a constant integer}}
}

void test_smfmac_f32_16x16x128_fp8_fp8(global float4* out, int4 a, int8 b, float4 c, int idx, int d)
{
  *out = __builtin_amdgcn_smfmac_f32_16x16x128_fp8_fp8(a, b, c, idx, d, 0); // expected-error{{argument to '__builtin_amdgcn_smfmac_f32_16x16x128_fp8_fp8' must be a constant integer}}
  *out = __builtin_amdgcn_smfmac_f32_16x16x128_fp8_fp8(a, b, c, idx, 0, d); // expected-error{{argument to '__builtin_amdgcn_smfmac_f32_16x16x128_fp8_fp8' must be a constant integer}}
}

void test_smfmac_f32_32x32x64_bf8_bf8(global float16* out, int4 a, int8 b, float16 c, int idx, int d)
{
  *out = __builtin_amdgcn_smfmac_f32_32x32x64_bf8_bf8(a, b, c, idx, d, 0); // expected-error{{argument to '__builtin_amdgcn_smfmac_f32_32x32x64_bf8_bf8' must be a constant integer}}
  *out = __builtin_amdgcn_smfmac_f32_32x32x64_bf8_bf8(a, b, c, idx, 0, d); // expected-error{{argument to '__builtin_amdgcn_smfmac_f32_32x32x64_bf8_bf8' must be a constant integer}}
}

void test_smfmac_f32_32x32x64_bf8_fp8(global float16* out, int4 a, int8 b, float16 c, int idx, int d)
{
  *out = __builtin_amdgcn_smfmac_f32_32x32x64_bf8_fp8(a, b, c, idx, d, 0); // expected-error{{argument to '__builtin_amdgcn_smfmac_f32_32x32x64_bf8_fp8' must be a constant integer}}
  *out = __builtin_amdgcn_smfmac_f32_32x32x64_bf8_fp8(a, b, c, idx, 0, d); // expected-error{{argument to '__builtin_amdgcn_smfmac_f32_32x32x64_bf8_fp8' must be a constant integer}}
}

void test_smfmac_f32_32x32x64_fp8_bf8(global float16* out, int4 a, int8 b, float16 c, int idx, int d)
{
  *out = __builtin_amdgcn_smfmac_f32_32x32x64_fp8_bf8(a, b, c, idx, d, 0); // expected-error{{argument to '__builtin_amdgcn_smfmac_f32_32x32x64_fp8_bf8' must be a constant integer}}
  *out = __builtin_amdgcn_smfmac_f32_32x32x64_fp8_bf8(a, b, c, idx, 0, d); // expected-error{{argument to '__builtin_amdgcn_smfmac_f32_32x32x64_fp8_bf8' must be a constant integer}}
}

void test_smfmac_f32_32x32x64_fp8_fp8(global float16* out, int4 a, int8 b, float16 c, int idx, int d)
{
  *out = __builtin_amdgcn_smfmac_f32_32x32x64_fp8_fp8(a, b, c, idx, d, 0); // expected-error{{argument to '__builtin_amdgcn_smfmac_f32_32x32x64_fp8_fp8' must be a constant integer}}
  *out = __builtin_amdgcn_smfmac_f32_32x32x64_fp8_fp8(a, b, c, idx, 0, d); // expected-error{{argument to '__builtin_amdgcn_smfmac_f32_32x32x64_fp8_fp8' must be a constant integer}}
}

void test_permlane16_swap(__global int* out, int old, int src, bool X) {
  *out = __builtin_amdgcn_permlane16_swap(old, src, X, false); // expected-error{{argument to '__builtin_amdgcn_permlane16_swap' must be a constant integer}}
  *out = __builtin_amdgcn_permlane16_swap(old, src, false, X); // expected-error{{argument to '__builtin_amdgcn_permlane16_swap' must be a constant integer}}
}

void test_permlane32_swap(__global int* out, int old, int src, bool X) {
  *out = __builtin_amdgcn_permlane32_swap(old, src, X, false); // expected-error{{argument to '__builtin_amdgcn_permlane32_swap' must be a constant integer}}
  *out = __builtin_amdgcn_permlane32_swap(old, src, false, X); // expected-error{{argument to '__builtin_amdgcn_permlane32_swap' must be a constant integer}}
}

void test_cvt_scalef32(global half2* out_v2f16, global float* out_f32, uint src, float scale, int index, bool X,
                       global short2* out_v2i16, float src0, float src1, global float2* out_v2f32,
                       half2 src0_v2f16, bfloat2 src0_v2bf16, float2 src0_v2f32, global uint* out, global bfloat2* out_v2bf16) {
  *out_v2f16 = __builtin_amdgcn_cvt_scalef32_f16_fp8(*out_v2f16, src, scale, index, X); // expected-error{{argument to '__builtin_amdgcn_cvt_scalef32_f16_fp8' must be a constant integer}}
  *out_f32 = __builtin_amdgcn_cvt_scalef32_f32_fp8(src, scale, index); // // expected-error{{argument to '__builtin_amdgcn_cvt_scalef32_f32_fp8' must be a constant integer}}
  *out_v2f16 = __builtin_amdgcn_cvt_scalef32_f16_bf8(*out_v2f16, src, scale, index, X); // expected-error{{argument to '__builtin_amdgcn_cvt_scalef32_f16_bf8' must be a constant integer}}
  *out_f32 = __builtin_amdgcn_cvt_scalef32_f32_bf8(src, scale, index); // expected-error{{argument to '__builtin_amdgcn_cvt_scalef32_f32_bf8' must be a constant integer}}
  *out_v2i16 = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(*out_v2i16, src0, src1, scale, X); // expected-error{{argument to '__builtin_amdgcn_cvt_scalef32_pk_fp8_f32' must be a constant integer}}
  *out_v2i16 = __builtin_amdgcn_cvt_scalef32_pk_bf8_f32(*out_v2i16, src0, src1, scale, X); // expected-error{{argument to '__builtin_amdgcn_cvt_scalef32_pk_bf8_f32' must be a constant integer}}
  *out_v2f32 = __builtin_amdgcn_cvt_scalef32_pk_f32_fp8(src, scale, X); // expected-error{{argument to '__builtin_amdgcn_cvt_scalef32_pk_f32_fp8' must be a constant integer}}
  *out_v2f32 = __builtin_amdgcn_cvt_scalef32_pk_f32_bf8(src, scale, X); // expected-error{{argument to '__builtin_amdgcn_cvt_scalef32_pk_f32_bf8' must be a constant integer}}
  *out_v2i16 = __builtin_amdgcn_cvt_scalef32_pk_fp8_f16(*out_v2i16, src0_v2f16, scale, X); // expected-error{{argument to '__builtin_amdgcn_cvt_scalef32_pk_fp8_f16' must be a constant integer}}
  *out_v2i16 = __builtin_amdgcn_cvt_scalef32_pk_fp8_bf16(*out_v2i16, src0_v2bf16, scale, X); // expected-error{{argument to '__builtin_amdgcn_cvt_scalef32_pk_fp8_bf16' must be a constant integer}}
  *out_v2i16 = __builtin_amdgcn_cvt_scalef32_pk_bf8_f16(*out_v2i16, src0_v2f16, scale, X); // expected-error{{argument to '__builtin_amdgcn_cvt_scalef32_pk_bf8_f16' must be a constant integer}}
  *out_v2i16 = __builtin_amdgcn_cvt_scalef32_pk_bf8_bf16(*out_v2i16, src0_v2bf16, scale, X); // expected-error{{argument to '__builtin_amdgcn_cvt_scalef32_pk_bf8_bf16' must be a constant integer}}
  *out_v2f32 = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(src, scale, index); // expected-error{{argument to '__builtin_amdgcn_cvt_scalef32_pk_f32_fp4' must be a constant integer}}
  *out = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(*out, src0, src1, scale, index); // expected-error{{argument to '__builtin_amdgcn_cvt_scalef32_pk_fp4_f32' must be a constant integer}}
  *out_v2f16 = __builtin_amdgcn_cvt_scalef32_pk_f16_fp4(src, scale, index); // expected-error{{argument to '__builtin_amdgcn_cvt_scalef32_pk_f16_fp4' must be a constant integer}}
  *out_v2bf16 = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4(src, scale, index); // expected-error{{argument to '__builtin_amdgcn_cvt_scalef32_pk_bf16_fp4' must be a constant integer}}
  *out_v2f16 = __builtin_amdgcn_cvt_scalef32_pk_f16_fp8(src, scale, X); // expected-error{{argument to '__builtin_amdgcn_cvt_scalef32_pk_f16_fp8' must be a constant integer}}
  *out_v2f16 = __builtin_amdgcn_cvt_scalef32_pk_f16_bf8(src, scale, X); // expected-error{{argument to '__builtin_amdgcn_cvt_scalef32_pk_f16_bf8' must be a constant integer}}
  *out_v2bf16 = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(src, scale, X); // expected-error{{argument to '__builtin_amdgcn_cvt_scalef32_pk_bf16_fp8' must be a constant integer}}
  *out_v2bf16 = __builtin_amdgcn_cvt_scalef32_pk_bf16_bf8(src, scale, X); // expected-error{{argument to '__builtin_amdgcn_cvt_scalef32_pk_bf16_bf8' must be a constant integer}}
  *out = __builtin_amdgcn_cvt_scalef32_pk_fp4_f16(*out, src0_v2f16, scale, index); // expected-error{{argument to '__builtin_amdgcn_cvt_scalef32_pk_fp4_f16' must be a constant integer}}
  *out = __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(*out, src0_v2bf16, scale, index); // expected-error{{argument to '__builtin_amdgcn_cvt_scalef32_pk_fp4_bf16' must be a constant integer}}
  *out = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f16(*out, src0_v2f16, 0, scale, index); // expected-error{{argument to '__builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f16' must be a constant integer}}
  *out = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_bf16(*out, src0_v2bf16, 0, scale, index); // expected-error{{argument to '__builtin_amdgcn_cvt_scalef32_sr_pk_fp4_bf16' must be a constant integer}}
  *out = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f32(*out, src0_v2f32, 0, scale, index); // expected-error{{argument to '__builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f32' must be a constant integer}}
  *out = __builtin_amdgcn_cvt_scalef32_sr_bf8_bf16(*out, src0, 0, scale, index); // expected-error{{argument to '__builtin_amdgcn_cvt_scalef32_sr_bf8_bf16' must be a constant integer}}
  *out = __builtin_amdgcn_cvt_scalef32_sr_bf8_f16(*out, src0, 0, scale, index); // expected-error{{argument to '__builtin_amdgcn_cvt_scalef32_sr_bf8_f16' must be a constant integer}}
  *out = __builtin_amdgcn_cvt_scalef32_sr_bf8_f32(*out, src0, 0, scale, index); // expected-error{{argument to '__builtin_amdgcn_cvt_scalef32_sr_bf8_f32' must be a constant integer}}
  *out = __builtin_amdgcn_cvt_scalef32_sr_fp8_bf16(*out, src0, 0, scale, index); // expected-error{{argument to '__builtin_amdgcn_cvt_scalef32_sr_fp8_bf16' must be a constant integer}}
  *out = __builtin_amdgcn_cvt_scalef32_sr_fp8_f16(*out, src0, 0, scale, index); // expected-error{{argument to '__builtin_amdgcn_cvt_scalef32_sr_fp8_f16' must be a constant integer}}
  *out = __builtin_amdgcn_cvt_scalef32_sr_fp8_f32(*out, src0, 0, scale, index); // expected-error{{argument to '__builtin_amdgcn_cvt_scalef32_sr_fp8_f32' must be a constant integer}}
  *out_v2bf16 = __builtin_amdgcn_cvt_sr_bf16_f32(*out_v2bf16, src0, src, X); // expected-error{{argument to '__builtin_amdgcn_cvt_sr_bf16_f32' must be a constant integer}}
  *out_v2f16 = __builtin_amdgcn_cvt_sr_f16_f32(*out_v2f16, src0, src, X); // expected-error{{argument to '__builtin_amdgcn_cvt_sr_f16_f32' must be a constant integer}}
}

void test_bitop3_args(global uint* out, uint a, uint b, uint c) {
  *out = __builtin_amdgcn_bitop3_b32(a, b, c, a); // expected-error {{argument to '__builtin_amdgcn_bitop3_b32' must be a constant integer}}
  *out = __builtin_amdgcn_bitop3_b16((ushort)a, (ushort)b, (ushort)c, a); // expected-error {{argument to '__builtin_amdgcn_bitop3_b16' must be a constant integer}}
}
