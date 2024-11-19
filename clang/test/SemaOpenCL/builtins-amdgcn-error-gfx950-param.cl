// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-- -target-cpu gfx950 -verify -S -o - %s

typedef float float4 __attribute__((ext_vector_type(4)));
typedef float float16 __attribute__((ext_vector_type(16)));
typedef half half8 __attribute__((ext_vector_type(8)));
typedef __bf16 bfloat8 __attribute__((ext_vector_type(8)));


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
