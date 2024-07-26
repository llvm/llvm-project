// RUN: %clang_cc1 -triple amdgcn-- -target-cpu gfx940 -verify -S -o - %s
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -verify -S -o - %s

typedef float  v2f   __attribute__((ext_vector_type(2)));
typedef float  v4f   __attribute__((ext_vector_type(4)));
typedef float  v16f  __attribute__((ext_vector_type(16)));
typedef int    v2i   __attribute__((ext_vector_type(2)));
typedef int    v4i   __attribute__((ext_vector_type(4)));
typedef int    v16i  __attribute__((ext_vector_type(16)));
typedef half   v4h   __attribute__((ext_vector_type(4)));
typedef half   v8h   __attribute__((ext_vector_type(8)));
typedef short  v4s   __attribute__((ext_vector_type(4)));
typedef short  v8s   __attribute__((ext_vector_type(8)));

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

void test_mfma_f32_16x16x32_bf8_bf8(global v4f* out, long a, long b, v4f c, int d)
{
  *out = __builtin_amdgcn_mfma_f32_16x16x32_bf8_bf8(a, b, c, d, 0, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_16x16x32_bf8_bf8' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_16x16x32_bf8_bf8(a, b, c, 0, d, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_16x16x32_bf8_bf8' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_16x16x32_bf8_bf8(a, b, c, 0, 0, d); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_16x16x32_bf8_bf8' must be a constant integer}}
}

void test_mfma_f32_16x16x32_bf8_fp8(global v4f* out, long a, long b, v4f c, int d)
{
  *out = __builtin_amdgcn_mfma_f32_16x16x32_bf8_fp8(a, b, c, d, 0, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_16x16x32_bf8_fp8' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_16x16x32_bf8_fp8(a, b, c, 0, d, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_16x16x32_bf8_fp8' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_16x16x32_bf8_fp8(a, b, c, 0, 0, d); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_16x16x32_bf8_fp8' must be a constant integer}}
}

void test_mfma_f32_16x16x32_fp8_bf8(global v4f* out, long a, long b, v4f c, int d)
{
  *out = __builtin_amdgcn_mfma_f32_16x16x32_fp8_bf8(a, b, c, d, 0, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_16x16x32_fp8_bf8' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_16x16x32_fp8_bf8(a, b, c, 0, d, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_16x16x32_fp8_bf8' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_16x16x32_fp8_bf8(a, b, c, 0, 0, d); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_16x16x32_fp8_bf8' must be a constant integer}}
}

void test_mfma_f32_16x16x32_fp8_fp8(global v4f* out, long a, long b, v4f c, int d)
{
  *out = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a, b, c, d, 0, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a, b, c, 0, d, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a, b, c, 0, 0, d); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8' must be a constant integer}}
}

void test_mfma_f32_32x32x16_bf8_bf8(global v16f* out, long a, long b, v16f c, int d)
{
  *out = __builtin_amdgcn_mfma_f32_32x32x16_bf8_bf8(a, b, c, d, 0, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_32x32x16_bf8_bf8' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_32x32x16_bf8_bf8(a, b, c, 0, d, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_32x32x16_bf8_bf8' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_32x32x16_bf8_bf8(a, b, c, 0, 0, d); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_32x32x16_bf8_bf8' must be a constant integer}}
}

void test_mfma_f32_32x32x16_bf8_fp8(global v16f* out, long a, long b, v16f c, int d)
{
  *out = __builtin_amdgcn_mfma_f32_32x32x16_bf8_fp8(a, b, c, d, 0, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_32x32x16_bf8_fp8' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_32x32x16_bf8_fp8(a, b, c, 0, d, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_32x32x16_bf8_fp8' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_32x32x16_bf8_fp8(a, b, c, 0, 0, d); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_32x32x16_bf8_fp8' must be a constant integer}}
}

void test_mfma_f32_32x32x16_fp8_bf8(global v16f* out, long a, long b, v16f c, int d)
{
  *out = __builtin_amdgcn_mfma_f32_32x32x16_fp8_bf8(a, b, c, d, 0, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_32x32x16_fp8_bf8' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_32x32x16_fp8_bf8(a, b, c, 0, d, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_32x32x16_fp8_bf8' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_32x32x16_fp8_bf8(a, b, c, 0, 0, d); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_32x32x16_fp8_bf8' must be a constant integer}}
}

void test_mfma_f32_32x32x16_fp8_fp8(global v16f* out, long a, long b, v16f c, int d)
{
  *out = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(a, b, c, d, 0, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(a, b, c, 0, d, 0); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8' must be a constant integer}}
  *out = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(a, b, c, 0, 0, d); // expected-error{{argument to '__builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8' must be a constant integer}}
}

void test_smfmac_f32_16x16x32_f16(global v4f* out, v4h a, v8h b, v4f c, int idx, int d)
{
  *out = __builtin_amdgcn_smfmac_f32_16x16x32_f16(a, b, c, idx, d, 0); // expected-error{{argument to '__builtin_amdgcn_smfmac_f32_16x16x32_f16' must be a constant integer}}
  *out = __builtin_amdgcn_smfmac_f32_16x16x32_f16(a, b, c, idx, 0, d); // expected-error{{argument to '__builtin_amdgcn_smfmac_f32_16x16x32_f16' must be a constant integer}}
}

void test_smfmac_f32_32x32x16_f16(global v16f* out, v4h a, v8h b, v16f c, int idx, int d)
{
  *out = __builtin_amdgcn_smfmac_f32_32x32x16_f16(a, b, c, idx, d, 0); // expected-error{{argument to '__builtin_amdgcn_smfmac_f32_32x32x16_f16' must be a constant integer}}
  *out = __builtin_amdgcn_smfmac_f32_32x32x16_f16(a, b, c, idx, 0, d); // expected-error{{argument to '__builtin_amdgcn_smfmac_f32_32x32x16_f16' must be a constant integer}}
}

void test_smfmac_f32_16x16x32_bf16(global v4f* out, v4s a, v8s b, v4f c, int idx, int d)
{
  *out = __builtin_amdgcn_smfmac_f32_16x16x32_bf16(a, b, c, idx, d, 0); // expected-error{{argument to '__builtin_amdgcn_smfmac_f32_16x16x32_bf16' must be a constant integer}}
  *out = __builtin_amdgcn_smfmac_f32_16x16x32_bf16(a, b, c, idx, 0, d); // expected-error{{argument to '__builtin_amdgcn_smfmac_f32_16x16x32_bf16' must be a constant integer}}
}

void test_smfmac_f32_32x32x16_bf16(global v16f* out, v4s a, v8s b, v16f c, int idx, int d)
{
  *out = __builtin_amdgcn_smfmac_f32_32x32x16_bf16(a, b, c, idx, d, 0); // expected-error{{argument to '__builtin_amdgcn_smfmac_f32_32x32x16_bf16' must be a constant integer}}
  *out = __builtin_amdgcn_smfmac_f32_32x32x16_bf16(a, b, c, idx, 0, d); // expected-error{{argument to '__builtin_amdgcn_smfmac_f32_32x32x16_bf16' must be a constant integer}}
}

void test_smfmac_i32_16x16x64_i8(global v4i* out, v2i a, v4i b, v4i c, int idx, int d)
{
  *out = __builtin_amdgcn_smfmac_i32_16x16x64_i8(a, b, c, idx, d, 0); // expected-error{{argument to '__builtin_amdgcn_smfmac_i32_16x16x64_i8' must be a constant integer}}
  *out = __builtin_amdgcn_smfmac_i32_16x16x64_i8(a, b, c, idx, 0, d); // expected-error{{argument to '__builtin_amdgcn_smfmac_i32_16x16x64_i8' must be a constant integer}}
}

void test_smfmac_i32_32x32x32_i8(global v16i* out, v2i a, v4i b, v16i c, int idx, int d)
{
  *out = __builtin_amdgcn_smfmac_i32_32x32x32_i8(a, b, c, idx, d, 0); // expected-error{{argument to '__builtin_amdgcn_smfmac_i32_32x32x32_i8' must be a constant integer}}
  *out = __builtin_amdgcn_smfmac_i32_32x32x32_i8(a, b, c, idx, 0, d); // expected-error{{argument to '__builtin_amdgcn_smfmac_i32_32x32x32_i8' must be a constant integer}}
}

void test_smfmac_f32_16x16x64_bf8_bf8(global v4f* out, v2i a, v4i b, v4f c, int idx, int d)
{
  *out = __builtin_amdgcn_smfmac_f32_16x16x64_bf8_bf8(a, b, c, idx, d, 0); // expected-error{{argument to '__builtin_amdgcn_smfmac_f32_16x16x64_bf8_bf8' must be a constant integer}}
  *out = __builtin_amdgcn_smfmac_f32_16x16x64_bf8_bf8(a, b, c, idx, 0, d); // expected-error{{argument to '__builtin_amdgcn_smfmac_f32_16x16x64_bf8_bf8' must be a constant integer}}
}

void test_smfmac_f32_16x16x64_bf8_fp8(global v4f* out, v2i a, v4i b, v4f c, int idx, int d)
{
  *out = __builtin_amdgcn_smfmac_f32_16x16x64_bf8_fp8(a, b, c, idx, d, 0); // expected-error{{argument to '__builtin_amdgcn_smfmac_f32_16x16x64_bf8_fp8' must be a constant integer}}
  *out = __builtin_amdgcn_smfmac_f32_16x16x64_bf8_fp8(a, b, c, idx, 0, d); // expected-error{{argument to '__builtin_amdgcn_smfmac_f32_16x16x64_bf8_fp8' must be a constant integer}}
}

void test_smfmac_f32_16x16x64_fp8_bf8(global v4f* out, v2i a, v4i b, v4f c, int idx, int d)
{
  *out = __builtin_amdgcn_smfmac_f32_16x16x64_fp8_bf8(a, b, c, idx, d, 0); // expected-error{{argument to '__builtin_amdgcn_smfmac_f32_16x16x64_fp8_bf8' must be a constant integer}}
  *out = __builtin_amdgcn_smfmac_f32_16x16x64_fp8_bf8(a, b, c, idx, 0, d); // expected-error{{argument to '__builtin_amdgcn_smfmac_f32_16x16x64_fp8_bf8' must be a constant integer}}
}

void test_smfmac_f32_16x16x64_fp8_fp8(global v4f* out, v2i a, v4i b, v4f c, int idx, int d)
{
  *out = __builtin_amdgcn_smfmac_f32_16x16x64_fp8_fp8(a, b, c, idx, d, 0); // expected-error{{argument to '__builtin_amdgcn_smfmac_f32_16x16x64_fp8_fp8' must be a constant integer}}
  *out = __builtin_amdgcn_smfmac_f32_16x16x64_fp8_fp8(a, b, c, idx, 0, d); // expected-error{{argument to '__builtin_amdgcn_smfmac_f32_16x16x64_fp8_fp8' must be a constant integer}}
}

void test_smfmac_f32_32x32x32_bf8_bf8(global v16f* out, v2i a, v4i b, v16f c, int idx, int d)
{
  *out = __builtin_amdgcn_smfmac_f32_32x32x32_bf8_bf8(a, b, c, idx, d, 0); // expected-error{{argument to '__builtin_amdgcn_smfmac_f32_32x32x32_bf8_bf8' must be a constant integer}}
  *out = __builtin_amdgcn_smfmac_f32_32x32x32_bf8_bf8(a, b, c, idx, 0, d); // expected-error{{argument to '__builtin_amdgcn_smfmac_f32_32x32x32_bf8_bf8' must be a constant integer}}
}

void test_smfmac_f32_32x32x32_bf8_fp8(global v16f* out, v2i a, v4i b, v16f c, int idx, int d)
{
  *out = __builtin_amdgcn_smfmac_f32_32x32x32_bf8_fp8(a, b, c, idx, d, 0); // expected-error{{argument to '__builtin_amdgcn_smfmac_f32_32x32x32_bf8_fp8' must be a constant integer}}
  *out = __builtin_amdgcn_smfmac_f32_32x32x32_bf8_fp8(a, b, c, idx, 0, d); // expected-error{{argument to '__builtin_amdgcn_smfmac_f32_32x32x32_bf8_fp8' must be a constant integer}}
}

void test_smfmac_f32_32x32x32_fp8_bf8(global v16f* out, v2i a, v4i b, v16f c, int idx, int d)
{
  *out = __builtin_amdgcn_smfmac_f32_32x32x32_fp8_bf8(a, b, c, idx, d, 0); // expected-error{{argument to '__builtin_amdgcn_smfmac_f32_32x32x32_fp8_bf8' must be a constant integer}}
  *out = __builtin_amdgcn_smfmac_f32_32x32x32_fp8_bf8(a, b, c, idx, 0, d); // expected-error{{argument to '__builtin_amdgcn_smfmac_f32_32x32x32_fp8_bf8' must be a constant integer}}
}

void test_smfmac_f32_32x32x32_fp8_fp8(global v16f* out, v2i a, v4i b, v16f c, int idx, int d)
{
  *out = __builtin_amdgcn_smfmac_f32_32x32x32_fp8_fp8(a, b, c, idx, d, 0); // expected-error{{argument to '__builtin_amdgcn_smfmac_f32_32x32x32_fp8_fp8' must be a constant integer}}
  *out = __builtin_amdgcn_smfmac_f32_32x32x32_fp8_fp8(a, b, c, idx, 0, d); // expected-error{{argument to '__builtin_amdgcn_smfmac_f32_32x32x32_fp8_fp8' must be a constant integer}}
}
