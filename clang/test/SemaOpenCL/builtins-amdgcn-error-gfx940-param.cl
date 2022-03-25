// RUN: %clang_cc1 -triple amdgcn-- -target-cpu gfx940 -verify -S -o - %s

typedef float  v2f   __attribute__((ext_vector_type(2)));
typedef float  v4f   __attribute__((ext_vector_type(4)));
typedef float  v16f  __attribute__((ext_vector_type(16)));
typedef int    v4i   __attribute__((ext_vector_type(4)));
typedef int    v16i  __attribute__((ext_vector_type(16)));

void test_mfma_i32_16x16x32i8(global v4i* out, long a, long b, v4i c, int d)
{
  *out = __builtin_amdgcn_mfma_i32_16x16x32_i8(a, b, c, d, 0, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_i32_16x16x32_i8' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_i32_16x16x32_i8(a, b, c, 0, d, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_i32_16x16x32_i8' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_i32_16x16x32_i8(a, b, c, 0, 0, d); // expected-error{{argument to '__builtin_amdgcn_mfma_i32_16x16x32_i8' must be a constant integer}}
}

void test_mfma_i32_32x32x16i8(global v16i* out, long a, long b, v16i c, int d)
{
  *out = __builtin_amdgcn_mfma_i32_32x32x16_i8(a, b, c, d, 0, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_i32_32x32x16_i8' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_i32_32x32x16_i8(a, b, c, 0, d, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_i32_32x32x16_i8' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_i32_32x32x16_i8(a, b, c, 0, 0, d); // expected-error{{argument to '__builtin_amdgcn_mfma_i32_32x32x16_i8' must be a constant integer}}
}

void test_mfma_f32_16x16x8xf32(global v4f* out, v2f a, v2f b, v4f c, int d)
{
  *out = __builtin_amdgcn_mfma_f32_16x16x8_xf32(a, b, c, d, 0, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_16x16x8_xf32' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_16x16x8_xf32(a, b, c, 0, d, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_16x16x8_xf32' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_16x16x8_xf32(a, b, c, 0, 0, d); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_16x16x8_xf32' must be a constant integer}}
}

void test_mfma_f32_32x32x4xf32(global v16f* out, v2f a, v2f b, v16f c, int d)
{
  *out = __builtin_amdgcn_mfma_f32_32x32x4_xf32(a, b, c, d, 0, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_32x32x4_xf32' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_32x32x4_xf32(a, b, c, 0, d, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_32x32x4_xf32' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_32x32x4_xf32(a, b, c, 0, 0, d); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_32x32x4_xf32' must be a constant integer}}
}
