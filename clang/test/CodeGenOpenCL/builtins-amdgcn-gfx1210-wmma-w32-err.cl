// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1100 -verify -S -emit-llvm -o - %s

typedef double v8d   __attribute__((ext_vector_type(8)));
typedef double v4d   __attribute__((ext_vector_type(4)));
typedef double v2d   __attribute__((ext_vector_type(2)));
typedef float  v8f   __attribute__((ext_vector_type(8)));
typedef float  v2f   __attribute__((ext_vector_type(2)));
typedef half   v32h   __attribute__((ext_vector_type(32)));
typedef half   v16h   __attribute__((ext_vector_type(16)));
typedef half   v8h   __attribute__((ext_vector_type(8)));
typedef short  v32s   __attribute__((ext_vector_type(32)));
typedef short  v16s   __attribute__((ext_vector_type(16)));
typedef short  v8s   __attribute__((ext_vector_type(8)));
typedef int    v16i   __attribute__((ext_vector_type(16)));
typedef int    v8i   __attribute__((ext_vector_type(8)));

void test_amdgcn_wmma_f64_16x16x4_f64_negA_err(global v8d* out, v2d a, v2d b, v8d c, bool negA)
{
  *out = __builtin_amdgcn_wmma_f64_16x16x4_f64(negA, a, 0, b, 0, c); // expected-error {{'__builtin_amdgcn_wmma_f64_16x16x4_f64' must be a constant integer}}
}

void test_amdgcn_wmma_f64_16x16x4_f64_negB(global v8d* out, v2d a, v2d b, v8d c, bool negB)
{
  *out = __builtin_amdgcn_wmma_f64_16x16x4_f64(0, a, negB, b, 0, c); // expected-error {{'__builtin_amdgcn_wmma_f64_16x16x4_f64' must be a constant integer}}
}

void test_amdgcn_wmma_f64_16x16x4_f64_modC_err(global v8d* out, v2d a, v2d b, v8d c, short modC)
{
  *out = __builtin_amdgcn_wmma_f64_16x16x4_f64(0, a, 0, b, modC, c); // expected-error {{'__builtin_amdgcn_wmma_f64_16x16x4_f64' must be a constant integer}}
}

void test_amdgcn_wmma_f64_16x16x8_f64_negA_err(global v8d* out, v4d a, v4d b, v8d c, bool negA)
{
  *out = __builtin_amdgcn_wmma_f64_16x16x8_f64(negA, a, 0, b, 0, c); // expected-error {{'__builtin_amdgcn_wmma_f64_16x16x8_f64' must be a constant integer}}
}

void test_amdgcn_wmma_f64_16x16x8_f64_negB_err(global v8d* out, v4d a, v4d b, v8d c, bool negB)
{
  *out = __builtin_amdgcn_wmma_f64_16x16x8_f64(0, a, negB, b, 0, c); // expected-error {{'__builtin_amdgcn_wmma_f64_16x16x8_f64' must be a constant integer}}
}

void test_amdgcn_wmma_f64_16x16x8_f64_modC_err(global v8d* out, v4d a, v4d b, v8d c, short modC)
{
  *out = __builtin_amdgcn_wmma_f64_16x16x8_f64(0, a, 0, b, modC, c); // expected-error {{'__builtin_amdgcn_wmma_f64_16x16x8_f64' must be a constant integer}}
}

void test_amdgcn_wmma_f32_16x16x4_f32_negA_err(global v8f* out, v2f a, v2f b, v8f c, bool negA)
{
  *out = __builtin_amdgcn_wmma_f32_16x16x4_f32(negA, a, 0, b, 0, c); // expected-error {{'__builtin_amdgcn_wmma_f32_16x16x4_f32' must be a constant integer}}
}

void test_amdgcn_wmma_f32_16x16x4_f32_negB_err(global v8f* out, v2f a, v2f b, v8f c, bool negB)
{
  *out = __builtin_amdgcn_wmma_f32_16x16x4_f32(0, a, negB, b, 0, c); // expected-error {{'__builtin_amdgcn_wmma_f32_16x16x4_f32' must be a constant integer}}
}

void test_amdgcn_wmma_f32_16x16x4_f32_modC_err(global v8f* out, v2f a, v2f b, v8f c, short modC)
{
  *out = __builtin_amdgcn_wmma_f32_16x16x4_f32(0, a, 0, b, modC, c); // expected-error {{'__builtin_amdgcn_wmma_f32_16x16x4_f32' must be a constant integer}}
}

void test_amdgcn_wmma_f32_16x16x32_bf16_negA_err(global v8f* out, v16s a, v16s b, v8f c, bool negA)
{
  *out = __builtin_amdgcn_wmma_f32_16x16x32_bf16(negA, a, 0, b, 0, c); // expected-error {{'__builtin_amdgcn_wmma_f32_16x16x32_bf16' must be a constant integer}}
}

void test_amdgcn_wmma_f32_16x16x32_bf16_negB_err(global v8f* out, v16s a, v16s b, v8f c, bool negB)
{
  *out = __builtin_amdgcn_wmma_f32_16x16x32_bf16(0, a, negB, b, 0, c); // expected-error {{'__builtin_amdgcn_wmma_f32_16x16x32_bf16' must be a constant integer}}
}

void test_amdgcn_wmma_f32_16x16x32_bf16_modC_err(global v8f* out, v16s a, v16s b, v8f c, short modC)
{
  *out = __builtin_amdgcn_wmma_f32_16x16x32_bf16(0, a, 0, b, modC, c); // expected-error {{'__builtin_amdgcn_wmma_f32_16x16x32_bf16' must be a constant integer}}
}

void test_amdgcn_wmma_f32_16x16x32_f16_negA_err(global v8f* out, v16h a, v16h b, v8f c, bool negA)
{
  *out = __builtin_amdgcn_wmma_f32_16x16x32_f16(negA, a, 0, b, 0, c); // expected-error {{'__builtin_amdgcn_wmma_f32_16x16x32_f16' must be a constant integer}}
}

void test_amdgcn_wmma_f32_16x16x32_f16_negB_err(global v8f* out, v16h a, v16h b, v8f c, bool negB)
{
  *out = __builtin_amdgcn_wmma_f32_16x16x32_f16(0, a, negB, b, 0, c); // expected-error {{'__builtin_amdgcn_wmma_f32_16x16x32_f16' must be a constant integer}}
}

void test_amdgcn_wmma_f32_16x16x32_f16_modC_err(global v8f* out, v16h a, v16h b, v8f c, short modC)
{
  *out = __builtin_amdgcn_wmma_f32_16x16x32_f16(0, a, 0, b, modC, c); // expected-error {{'__builtin_amdgcn_wmma_f32_16x16x32_f16' must be a constant integer}}
}

void test_amdgcn_wmma_f16_16x16x32_f16_negA_err(global v8h* out, v16h a, v16h b, v8h c, bool negA)
{
  *out = __builtin_amdgcn_wmma_f16_16x16x32_f16(negA, a, 0, b, 0, c); // expected-error {{'__builtin_amdgcn_wmma_f16_16x16x32_f16' must be a constant integer}}
}

void test_amdgcn_wmma_f16_16x16x32_f16_negB_err(global v8h* out, v16h a, v16h b, v8h c, bool negB)
{
  *out = __builtin_amdgcn_wmma_f16_16x16x32_f16(0, a, negB, b, 0, c); // expected-error {{'__builtin_amdgcn_wmma_f16_16x16x32_f16' must be a constant integer}}
}

void test_amdgcn_wmma_f16_16x16x32_f16_modC_err(global v8h* out, v16h a, v16h b, v8h c, short modC)
{
  *out = __builtin_amdgcn_wmma_f16_16x16x32_f16(0, a, 0, b, modC, c); // expected-error {{'__builtin_amdgcn_wmma_f16_16x16x32_f16' must be a constant integer}}
}

void test_amdgcn_wmma_bf16_16x16x32_bf16_negA_err(global v8s* out, v16s a, v16s b, v8s c, bool negA)
{
  *out = __builtin_amdgcn_wmma_bf16_16x16x32_bf16(negA, a, 0, b, 0, c); // expected-error {{'__builtin_amdgcn_wmma_bf16_16x16x32_bf16' must be a constant integer}}
}

void test_amdgcn_wmma_bf16_16x16x32_bf16_negB_err(global v8s* out, v16s a, v16s b, v8s c, bool negB)
{
  *out = __builtin_amdgcn_wmma_bf16_16x16x32_bf16(0, a, negB, b, 0, c); // expected-error {{'__builtin_amdgcn_wmma_bf16_16x16x32_bf16' must be a constant integer}}
}

void test_amdgcn_wmma_bf16_16x16x32_bf16_modC_err(global v8s* out, v16s a, v16s b, v8s c, short modC)
{
  *out = __builtin_amdgcn_wmma_bf16_16x16x32_bf16(0, a, 0, b, modC, c); // expected-error {{'__builtin_amdgcn_wmma_bf16_16x16x32_bf16' must be a constant integer}}
}

void test_amdgcn_wmma_bf16f32_16x16x32_bf16_negA_err(global v8s* out, v16s a, v16s b, v8f c, bool negA)
{
  *out = __builtin_amdgcn_wmma_bf16f32_16x16x32_bf16(negA, a, 0, b, 0, c); // expected-error {{'__builtin_amdgcn_wmma_bf16f32_16x16x32_bf16' must be a constant integer}}
}

void test_amdgcn_wmma_bf16f32_16x16x32_bf16_negB_err(global v8s* out, v16s a, v16s b, v8f c, bool negB)
{
  *out = __builtin_amdgcn_wmma_bf16f32_16x16x32_bf16(0, a, negB, b, 0, c); // expected-error {{'__builtin_amdgcn_wmma_bf16f32_16x16x32_bf16' must be a constant integer}}
}

void test_amdgcn_wmma_bf16f32_16x16x32_bf16_modC_err(global v8s* out, v16s a, v16s b, v8f c, short modC)
{
  *out = __builtin_amdgcn_wmma_bf16f32_16x16x32_bf16(0, a, 0, b, modC, c); // expected-error {{'__builtin_amdgcn_wmma_bf16f32_16x16x32_bf16' must be a constant integer}}
}

void test_amdgcn_wmma_f32_16x16x64_fp8_fp8_modC_err(global v8f* out, v8i a, v8i b, v8f c, short modC)
{
  *out = __builtin_amdgcn_wmma_f32_16x16x64_fp8_fp8(a, b, modC, c); // expected-error {{'__builtin_amdgcn_wmma_f32_16x16x64_fp8_fp8' must be a constant integer}}
}

void test_amdgcn_wmma_f32_16x16x64_fp8_bf8_modC_err(global v8f* out, v8i a, v8i b, v8f c, short modC)
{
  *out = __builtin_amdgcn_wmma_f32_16x16x64_fp8_bf8(a, b, modC, c); // expected-error {{'__builtin_amdgcn_wmma_f32_16x16x64_fp8_bf8' must be a constant integer}}
}

void test_amdgcn_wmma_f32_16x16x64_bf8_fp8_modC_err(global v8f* out, v8i a, v8i b, v8f c, short modC)
{
  *out = __builtin_amdgcn_wmma_f32_16x16x64_bf8_fp8(a, b, modC, c); // expected-error {{'__builtin_amdgcn_wmma_f32_16x16x64_bf8_fp8' must be a constant integer}}
}

void test_amdgcn_wmma_f32_16x16x64_bf8_bf8_modC_err(global v8f* out, v8i a, v8i b, v8f c, short modC)
{
  *out = __builtin_amdgcn_wmma_f32_16x16x64_bf8_bf8(a, b, modC, c); // expected-error {{'__builtin_amdgcn_wmma_f32_16x16x64_bf8_bf8' must be a constant integer}}
}

void test_amdgcn_wmma_f16_16x16x64_fp8_fp8_modC_err(global v8h* out, v8i a, v8i b, v8h c, short modC)
{
  *out = __builtin_amdgcn_wmma_f16_16x16x64_fp8_fp8(a, b, modC, c); // expected-error {{'__builtin_amdgcn_wmma_f16_16x16x64_fp8_fp8' must be a constant integer}}
}

void test_amdgcn_wmma_f16_16x16x64_fp8_bf8_modC_err(global v8h* out, v8i a, v8i b, v8h c, short modC)
{
  *out = __builtin_amdgcn_wmma_f16_16x16x64_fp8_bf8(a, b, modC, c); // expected-error {{'__builtin_amdgcn_wmma_f16_16x16x64_fp8_bf8' must be a constant integer}}
}

void test_amdgcn_wmma_f16_16x16x64_bf8_fp8_modC_err(global v8h* out, v8i a, v8i b, v8h c, short modC)
{
  *out = __builtin_amdgcn_wmma_f16_16x16x64_bf8_fp8(a, b, modC, c); // expected-error {{'__builtin_amdgcn_wmma_f16_16x16x64_bf8_fp8' must be a constant integer}}
}

void test_amdgcn_wmma_f16_16x16x64_bf8_bf8_modC_err(global v8h* out, v8i a, v8i b, v8h c, short modC)
{
  *out = __builtin_amdgcn_wmma_f16_16x16x64_bf8_bf8(a, b, modC, c); // expected-error {{'__builtin_amdgcn_wmma_f16_16x16x64_bf8_bf8' must be a constant integer}}
}

void test_amdgcn_wmma_i32_16x16x64_iu8_signA_err(global v8i* out, v8i a, v8i b, v8i c, bool signA)
{
  *out = __builtin_amdgcn_wmma_i32_16x16x64_iu8(signA, a, 0, b, c); // expected-error {{'__builtin_amdgcn_wmma_i32_16x16x64_iu8' must be a constant integer}}
}

void test_amdgcn_wmma_i32_16x16x64_iu8_signB_err(global v8i* out, v8i a, v8i b, v8i c, bool signB)
{
  *out = __builtin_amdgcn_wmma_i32_16x16x64_iu8(0, a, signB, b, c); // expected-error {{'__builtin_amdgcn_wmma_i32_16x16x64_iu8' must be a constant integer}}
}

void test_amdgcn_wmma_i32_16x16x128_iu4_signA_err(global v8i* out, v8i a, v8i b, v8i c, bool signA)
{
  *out = __builtin_amdgcn_wmma_i32_16x16x128_iu4(signA, a, 0, b, c); // expected-error {{'__builtin_amdgcn_wmma_i32_16x16x128_iu4' must be a constant integer}}
}

void test_amdgcn_wmma_i32_16x16x128_iu4_signB_err(global v8i* out, v8i a, v8i b, v8i c, bool signB)
{
  *out = __builtin_amdgcn_wmma_i32_16x16x128_iu4(0, a, signB, b, c); // expected-error {{'__builtin_amdgcn_wmma_i32_16x16x128_iu4' must be a constant integer}}
}

void test_amdgcn_swmmac_f32_16x16x64_f16_negA_err(global v8f* out, v16h a, v32h b, v8f c, short index, bool negA)
{
  *out = __builtin_amdgcn_swmmac_f32_16x16x64_f16(negA, a, 0, b, c, index); // expected-error {{'__builtin_amdgcn_swmmac_f32_16x16x64_f16' must be a constant integer}}
}

void test_amdgcn_swmmac_f32_16x16x64_f16_f16_negB_err(global v8f* out, v16h a, v32h b, v8f c, short index, bool negB)
{
  *out = __builtin_amdgcn_swmmac_f32_16x16x64_f16(0, a, negB, b, c, index); // expected-error {{'__builtin_amdgcn_swmmac_f32_16x16x64_f16' must be a constant integer}}
}

void test_amdgcn_swmmac_f32_16x16x64_bf16_f16_negA_err(global v8f* out, v16s a, v32s b, v8f c, short index, bool negA)
{
  *out = __builtin_amdgcn_swmmac_f32_16x16x64_bf16(negA, a, 0, b, c, index); // expected-error {{'__builtin_amdgcn_swmmac_f32_16x16x64_bf16' must be a constant integer}}
}

void test_amdgcn_swmmac_f32_16x16x64_bf16_f16_negB_err(global v8f* out, v16s a, v32s b, v8f c, short index, bool negB)
{
  *out = __builtin_amdgcn_swmmac_f32_16x16x64_bf16(0, a, negB, b, c, index); // expected-error {{'__builtin_amdgcn_swmmac_f32_16x16x64_bf16' must be a constant integer}}
}

void test_amdgcn_swmmac_f16_16x16x64_f16_f16_negA_err(global v8h* out, v16h a, v32h b, v8h c, short index, bool negA)
{
  *out = __builtin_amdgcn_swmmac_f16_16x16x64_f16(negA, a, 0, b, c, index); // expected-error {{'__builtin_amdgcn_swmmac_f16_16x16x64_f16' must be a constant integer}}
}

void test_amdgcn_swmmac_f16_16x16x64_f16_f16_negB_err(global v8h* out, v16h a, v32h b, v8h c, short index, bool negB)
{
  *out = __builtin_amdgcn_swmmac_f16_16x16x64_f16(0, a, negB, b, c, index); // expected-error {{'__builtin_amdgcn_swmmac_f16_16x16x64_f16' must be a constant integer}}
}

void test_amdgcn_swmmac_bf16_16x16x64_bf16_negA_err(global v8s* out, v16s a, v32s b, v8s c, short index, bool negA)
{
  *out = __builtin_amdgcn_swmmac_bf16_16x16x64_bf16(negA, a, 0, b, c, index); // expected-error {{'__builtin_amdgcn_swmmac_bf16_16x16x64_bf16' must be a constant integer}}
}

void test_amdgcn_swmmac_bf16_16x16x64_bf16_f16_negB_err(global v8s* out, v16s a, v32s b, v8s c, short index, bool negB)
{
  *out = __builtin_amdgcn_swmmac_bf16_16x16x64_bf16(0, a, negB, b, c, index); // expected-error {{'__builtin_amdgcn_swmmac_bf16_16x16x64_bf16' must be a constant integer}}
}

void test_amdgcn_swmmac_bf16f32_16x16x64_bf16_negA_err(global v8f* out, v16s a, v32s b, v8f c, short index, bool negA)
{
  *out = __builtin_amdgcn_swmmac_bf16f32_16x16x64_bf16(negA, a, 0, b, c, index); // expected-error {{'__builtin_amdgcn_swmmac_bf16f32_16x16x64_bf16' must be a constant integer}}
}

void test_amdgcn_swmmac_bf16f32_16x16x64_bf16_negB_err(global v8f* out, v16s a, v32s b, v8f c, short index, bool negB)
{
  *out = __builtin_amdgcn_swmmac_bf16f32_16x16x64_bf16(0, a, negB, b, c, index); // expected-error {{'__builtin_amdgcn_swmmac_bf16f32_16x16x64_bf16' must be a constant integer}}
}

void test_amdgcn_swmmac_i32_16x16x128_iu8_signA_err(global v8i* out, v8i a, v16i b, v8i c, short index, bool signA)
{
  *out = __builtin_amdgcn_swmmac_i32_16x16x128_iu8(signA, a, true, b, c, index); // expected-error {{'__builtin_amdgcn_swmmac_i32_16x16x128_iu8' must be a constant integer}}
}

void test_amdgcn_swmmac_i32_16x16x128_iu8_signB_err(global v8i* out, v8i a, v16i b, v8i c, short index, bool signB)
{
  *out = __builtin_amdgcn_swmmac_i32_16x16x128_iu8(true, a, signB, b, c, index); // expected-error {{'__builtin_amdgcn_swmmac_i32_16x16x128_iu8' must be a constant integer}}
}

void test_amdgcn_swmmac_i32_16x16x256_iu4_signA_err(global v8i* out, v8i a, v16i b, v8i c, short index, bool signA)
{
  *out = __builtin_amdgcn_swmmac_i32_16x16x256_iu4(signA, a, true, b, c, index); // expected-error {{'__builtin_amdgcn_swmmac_i32_16x16x256_iu4' must be a constant integer}}
}

void test_amdgcn_swmmac_i32_16x16x256_iu4_signB_err(global v8i* out, v8i a, v16i b, v8i c, short index, bool signB)
{
  *out = __builtin_amdgcn_swmmac_i32_16x16x256_iu4(true, a, signB, b, c, index); // expected-error {{'__builtin_amdgcn_swmmac_i32_16x16x256_iu4' must be a constant integer}}
}
