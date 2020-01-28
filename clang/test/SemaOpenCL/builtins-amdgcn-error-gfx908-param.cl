// RUN: %clang_cc1 -triple amdgcn-- -target-cpu gfx908 -verify -S -o - %s

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


void test_mfma_f32_32x32x1f32(global v32f* out, float a, float b, v32f c, int d)
{
  *out = __builtin_amdgcn_mfma_f32_32x32x1f32(a, b, c, d, 0, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_32x32x1f32' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_32x32x1f32(a, b, c, 0, d, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_32x32x1f32' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_32x32x1f32(a, b, c, 0, 0, d); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_32x32x1f32' must be a constant integer}}
}

void test_mfma_f32_16x16x1f32(global v16f* out, float a, float b, v16f c, int d)
{
  *out = __builtin_amdgcn_mfma_f32_16x16x1f32(a, b, c, d, 0, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_16x16x1f32' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_16x16x1f32(a, b, c, 0, d, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_16x16x1f32' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_16x16x1f32(a, b, c, 0, 0, d); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_16x16x1f32' must be a constant integer}}
}

void test_mfma_f32_4x4x1f32(global v4f* out, float a, float b, v4f c, int d)
{
  *out = __builtin_amdgcn_mfma_f32_4x4x1f32(a, b, c, d, 0, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_4x4x1f32' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_4x4x1f32(a, b, c, 0, d, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_4x4x1f32' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_4x4x1f32(a, b, c, 0, 0, d); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_4x4x1f32' must be a constant integer}}
}

void test_mfma_f32_32x32x2f32(global v16f* out, float a, float b, v16f c, int d)
{
  *out = __builtin_amdgcn_mfma_f32_32x32x2f32(a, b, c, d, 0, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_32x32x2f32' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_32x32x2f32(a, b, c, 0, d, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_32x32x2f32' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_32x32x2f32(a, b, c, 0, 0, d); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_32x32x2f32' must be a constant integer}}
}

void test_mfma_f32_16x16x4f32(global v4f* out, float a, float b, v4f c, int d)
{
  *out = __builtin_amdgcn_mfma_f32_16x16x4f32(a, b, c, d, 0, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_16x16x4f32' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_16x16x4f32(a, b, c, 0, d, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_16x16x4f32' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_16x16x4f32(a, b, c, 0, 0, d); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_16x16x4f32' must be a constant integer}}
}

void test_mfma_f32_32x32x4f16(global v32f* out, v4h a, v4h b, v32f c, int d)
{
  *out = __builtin_amdgcn_mfma_f32_32x32x4f16(a, b, c, d, 0, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_32x32x4f16' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_32x32x4f16(a, b, c, 0, d, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_32x32x4f16' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_32x32x4f16(a, b, c, 0, 0, d); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_32x32x4f16' must be a constant integer}}
}

void test_mfma_f32_16x16x4f16(global v16f* out, v4h a, v4h b, v16f c, int d)
{
  *out = __builtin_amdgcn_mfma_f32_16x16x4f16(a, b, c, d, 0, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_16x16x4f16' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_16x16x4f16(a, b, c, 0, d, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_16x16x4f16' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_16x16x4f16(a, b, c, 0, 0, d); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_16x16x4f16' must be a constant integer}}
}

void test_mfma_f32_4x4x4f16(global v4f* out, v4h a, v4h b, v4f c, int d)
{
  *out = __builtin_amdgcn_mfma_f32_4x4x4f16(a, b, c, d, 0, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_4x4x4f16' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_4x4x4f16(a, b, c, 0, d, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_4x4x4f16' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_4x4x4f16(a, b, c, 0, 0, d); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_4x4x4f16' must be a constant integer}}
}

void test_mfma_f32_32x32x8f16(global v16f* out, v4h a, v4h b, v16f c, int d)
{
  *out = __builtin_amdgcn_mfma_f32_32x32x8f16(a, b, c, d, 0, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_32x32x8f16' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_32x32x8f16(a, b, c, 0, d, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_32x32x8f16' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_32x32x8f16(a, b, c, 0, 0, d); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_32x32x8f16' must be a constant integer}}
}

void test_mfma_f32_16x16x16f16(global v4f* out, v4h a, v4h b, v4f c, int d)
{
  *out = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c, d, 0, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_16x16x16f16' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c, 0, d, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_16x16x16f16' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c, 0, 0, d); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_16x16x16f16' must be a constant integer}}
}

void test_mfma_i32_32x32x4i8(global v32i* out, int a, int b, v32i c, int d)
{
  *out = __builtin_amdgcn_mfma_i32_32x32x4i8(a, b, c, d, 0, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_i32_32x32x4i8' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_i32_32x32x4i8(a, b, c, 0, d, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_i32_32x32x4i8' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_i32_32x32x4i8(a, b, c, 0, 0, d); // expected-error{{argument to '__builtin_amdgcn_mfma_i32_32x32x4i8' must be a constant integer}}
}

void test_mfma_i32_16x16x4i8(global v16i* out, int a, int b, v16i c, int d)
{
  *out = __builtin_amdgcn_mfma_i32_16x16x4i8(a, b, c, d, 0, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_i32_16x16x4i8' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_i32_16x16x4i8(a, b, c, 0, d, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_i32_16x16x4i8' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_i32_16x16x4i8(a, b, c, 0, 0, d); // expected-error{{argument to '__builtin_amdgcn_mfma_i32_16x16x4i8' must be a constant integer}}
}

void test_mfma_i32_4x4x4i8(global v4i* out, int a, int b, v4i c, int d)
{
  *out = __builtin_amdgcn_mfma_i32_4x4x4i8(a, b, c, d, 0, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_i32_4x4x4i8' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_i32_4x4x4i8(a, b, c, 0, d, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_i32_4x4x4i8' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_i32_4x4x4i8(a, b, c, 0, 0, d); // expected-error{{argument to '__builtin_amdgcn_mfma_i32_4x4x4i8' must be a constant integer}}
}

void test_mfma_i32_32x32x8i8(global v16i* out, int a, int b, v16i c, int d)
{
  *out = __builtin_amdgcn_mfma_i32_32x32x8i8(a, b, c, d, 0, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_i32_32x32x8i8' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_i32_32x32x8i8(a, b, c, 0, d, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_i32_32x32x8i8' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_i32_32x32x8i8(a, b, c, 0, 0, d); // expected-error{{argument to '__builtin_amdgcn_mfma_i32_32x32x8i8' must be a constant integer}}
}

void test_mfma_i32_16x16x16i8(global v4i* out, int a, int b, v4i c, int d)
{
  *out = __builtin_amdgcn_mfma_i32_16x16x16i8(a, b, c, d, 0, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_i32_16x16x16i8' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_i32_16x16x16i8(a, b, c, 0, d, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_i32_16x16x16i8' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_i32_16x16x16i8(a, b, c, 0, 0, d); // expected-error{{argument to '__builtin_amdgcn_mfma_i32_16x16x16i8' must be a constant integer}}
}

void test_mfma_f32_32x32x2bf16(global v32f* out, v2s a, v2s b, v32f c, int d)
{
  *out = __builtin_amdgcn_mfma_f32_32x32x2bf16(a, b, c, d, 0, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_32x32x2bf16' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_32x32x2bf16(a, b, c, 0, d, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_32x32x2bf16' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_32x32x2bf16(a, b, c, 0, 0, d); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_32x32x2bf16' must be a constant integer}}
}

void test_mfma_f32_16x16x2bf16(global v16f* out, v2s a, v2s b, v16f c, int d)
{
  *out = __builtin_amdgcn_mfma_f32_16x16x2bf16(a, b, c, d, 0, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_16x16x2bf16' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_16x16x2bf16(a, b, c, 0, d, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_16x16x2bf16' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_16x16x2bf16(a, b, c, 0, 0, d); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_16x16x2bf16' must be a constant integer}}
}

void test_mfma_f32_4x4x2bf16(global v4f* out, v2s a, v2s b, v4f c, int d)
{
  *out = __builtin_amdgcn_mfma_f32_4x4x2bf16(a, b, c, d, 0, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_4x4x2bf16' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_4x4x2bf16(a, b, c, 0, d, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_4x4x2bf16' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_4x4x2bf16(a, b, c, 0, 0, d); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_4x4x2bf16' must be a constant integer}}
}

void test_mfma_f32_32x32x4bf16(global v16f* out, v2s a, v2s b, v16f c, int d)
{
  *out = __builtin_amdgcn_mfma_f32_32x32x4bf16(a, b, c, d, 0, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_32x32x4bf16' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_32x32x4bf16(a, b, c, 0, d, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_32x32x4bf16' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_32x32x4bf16(a, b, c, 0, 0, d); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_32x32x4bf16' must be a constant integer}}
}

void test_mfma_f32_16x16x8bf16(global v4f* out, v2s a, v2s b, v4f c, int d)
{
  *out = __builtin_amdgcn_mfma_f32_16x16x8bf16(a, b, c, d, 0, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_16x16x8bf16' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_16x16x8bf16(a, b, c, 0, d, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_16x16x8bf16' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_16x16x8bf16(a, b, c, 0, 0, d); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_16x16x8bf16' must be a constant integer}}
}
