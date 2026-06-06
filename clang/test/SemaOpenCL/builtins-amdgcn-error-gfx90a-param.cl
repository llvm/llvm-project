// RUN: %clang_cc1 -triple amdgcn-- -target-cpu gfx90a -verify -S -o - %s
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -verify -S -o - %s

#pragma OPENCL EXTENSION cl_khr_fp64:enable

typedef float  v4f   __attribute__((ext_vector_type(4)));
typedef float  v16f  __attribute__((ext_vector_type(16)));
typedef float  v32f  __attribute__((ext_vector_type(32)));
typedef half   v4h   __attribute__((ext_vector_type(4)));
typedef half   v16h  __attribute__((ext_vector_type(16)));
typedef half   v32h  __attribute__((ext_vector_type(32)));
typedef int    v4i   __attribute__((ext_vector_type(4)));
typedef int    v16i  __attribute__((ext_vector_type(16)));
typedef int    v32i  __attribute__((ext_vector_type(32)));
typedef short  v2s   __attribute__((ext_vector_type(2)));
typedef short  v4s   __attribute__((ext_vector_type(4)));
typedef short  v16s  __attribute__((ext_vector_type(16)));
typedef short  v32s  __attribute__((ext_vector_type(32)));
typedef double v4d   __attribute__((ext_vector_type(4)));

void test_mfma_f32_32x32x4bf16_1k(global v32f* out, v4s a, v4s b, v32f c, int d)
{
  *out = __builtin_amdgcn_mfma_f32_32x32x4bf16_1k(a, b, c, d, 0, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_32x32x4bf16_1k' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_32x32x4bf16_1k(a, b, c, 0, d, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_32x32x4bf16_1k' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_32x32x4bf16_1k(a, b, c, 0, 0, d); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_32x32x4bf16_1k' must be a constant integer}}
}

void test_mfma_f32_16x16x4bf16_1k(global v16f* out, v4s a, v4s b, v16f c, int d)
{
  *out = __builtin_amdgcn_mfma_f32_16x16x4bf16_1k(a, b, c, d, 0, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_16x16x4bf16_1k' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_16x16x4bf16_1k(a, b, c, 0, d, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_16x16x4bf16_1k' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_16x16x4bf16_1k(a, b, c, 0, 0, d); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_16x16x4bf16_1k' must be a constant integer}}
}

void test_mfma_f32_4x4x4bf16_1k(global v4f* out, v4s a, v4s b, v4f c, int d)
{
  *out = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(a, b, c, d, 0, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_4x4x4bf16_1k' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(a, b, c, 0, d, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_4x4x4bf16_1k' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(a, b, c, 0, 0, d); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_4x4x4bf16_1k' must be a constant integer}}
}

void test_mfma_f32_32x32x8bf16_1k(global v16f* out, v4s a, v4s b, v16f c, int d)
{
  *out = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a, b, c, d, 0, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_32x32x8bf16_1k' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a, b, c, 0, d, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_32x32x8bf16_1k' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a, b, c, 0, 0, d); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_32x32x8bf16_1k' must be a constant integer}}
}

void test_mfma_f32_16x16x16bf16_1k(global v4f* out, v4s a, v4s b, v4f c, int d)
{
  *out = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, c, d, 0, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_16x16x16bf16_1k' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, c, 0, d, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_16x16x16bf16_1k' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, c, 0, 0, d); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_16x16x16bf16_1k' must be a constant integer}}
}

void test_mfma_f64_16x16x4f64(global v4d* out, double a, double b, v4d c, int d)
{
  *out = __builtin_amdgcn_mfma_f64_16x16x4f64(a, b, c, d, 0, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f64_16x16x4f64' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f64_16x16x4f64(a, b, c, 0, d, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f64_16x16x4f64' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f64_16x16x4f64(a, b, c, 0, 0, d); // expected-error{{argument to '__builtin_amdgcn_mfma_f64_16x16x4f64' must be a constant integer}}
}

void test_mfma_f64_4x4x4f64(global double* out, double a, double b, double c, int d)
{
  *out = __builtin_amdgcn_mfma_f64_4x4x4f64(a, b, c, d, 0, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f64_4x4x4f64' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f64_4x4x4f64(a, b, c, 0, d, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f64_4x4x4f64' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f64_4x4x4f64(a, b, c, 0, 0, d); // expected-error{{argument to '__builtin_amdgcn_mfma_f64_4x4x4f64' must be a constant integer}}
}
