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
