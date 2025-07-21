// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1250 -verify -emit-llvm -o - %s

typedef float  v16f  __attribute__((ext_vector_type(16)));
typedef float  v8f   __attribute__((ext_vector_type(8)));
typedef float  v2f   __attribute__((ext_vector_type(2)));
typedef half   v8h   __attribute__((ext_vector_type(8)));
typedef half   v16h  __attribute__((ext_vector_type(16)));
typedef half   v32h   __attribute__((ext_vector_type(32)));
typedef __bf16 v32bf16 __attribute__((ext_vector_type(32)));
typedef __bf16 v16bf16 __attribute__((ext_vector_type(16)));
typedef __bf16 v8bf16 __attribute__((ext_vector_type(8)));
typedef int    v16i   __attribute__((ext_vector_type(16)));
typedef int    v8i   __attribute__((ext_vector_type(8)));

void test_amdgcn_wmma_f32_16x16x4_f32(global v8f* out, v2f a, v2f b, v8f c, int mod)
{
  *out = __builtin_amdgcn_wmma_f32_16x16x4_f32(mod, a, 0, b, 0, c, false, false); // expected-error {{'__builtin_amdgcn_wmma_f32_16x16x4_f32' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_f32_16x16x4_f32(0, a, mod, b, 0, c, false, false); // expected-error {{'__builtin_amdgcn_wmma_f32_16x16x4_f32' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_f32_16x16x4_f32(0, a, 0, b, mod, c, false, false); // expected-error {{'__builtin_amdgcn_wmma_f32_16x16x4_f32' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_f32_16x16x4_f32(0, a, 0, b, 0, c, mod, false); // expected-error {{'__builtin_amdgcn_wmma_f32_16x16x4_f32' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_f32_16x16x4_f32(0, a, 0, b, 0, c, false, mod); // expected-error {{'__builtin_amdgcn_wmma_f32_16x16x4_f32' must be a constant integer}}
}

void test_amdgcn_wmma_f32_16x16x32_bf16(global v8f* out, v16bf16 a, v16bf16 b, v8f c, int mod)
{
  *out = __builtin_amdgcn_wmma_f32_16x16x32_bf16(mod, a, 0, b, 0, c, false, false); // expected-error {{'__builtin_amdgcn_wmma_f32_16x16x32_bf16' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_f32_16x16x32_bf16(0, a, mod, b, 0, c, false, false); // expected-error {{'__builtin_amdgcn_wmma_f32_16x16x32_bf16' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_f32_16x16x32_bf16(0, a, 0, b, mod, c, false, false); // expected-error {{'__builtin_amdgcn_wmma_f32_16x16x32_bf16' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_f32_16x16x32_bf16(0, a, 0, b, 0, c, mod, false); // expected-error {{'__builtin_amdgcn_wmma_f32_16x16x32_bf16' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_f32_16x16x32_bf16(0, a, 0, b, 0, c, false, mod); // expected-error {{'__builtin_amdgcn_wmma_f32_16x16x32_bf16' must be a constant integer}}
}

void test_amdgcn_wmma_bf16_16x16x32_bf16(global v8bf16* out, v16bf16 a, v16bf16 b, v8bf16 c, int mod)
{
  *out = __builtin_amdgcn_wmma_bf16_16x16x32_bf16(mod, a, 0, b, 0, c, false, false); // expected-error {{'__builtin_amdgcn_wmma_bf16_16x16x32_bf16' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_bf16_16x16x32_bf16(0, a, mod, b, 0, c, false, false); // expected-error {{'__builtin_amdgcn_wmma_bf16_16x16x32_bf16' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_bf16_16x16x32_bf16(0, a, 0, b, mod, c, false, false); // expected-error {{'__builtin_amdgcn_wmma_bf16_16x16x32_bf16' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_bf16_16x16x32_bf16(0, a, 0, b, 0, c, mod, false); // expected-error {{'__builtin_amdgcn_wmma_bf16_16x16x32_bf16' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_bf16_16x16x32_bf16(0, a, 0, b, 0, c, false, mod); // expected-error {{'__builtin_amdgcn_wmma_bf16_16x16x32_bf16' must be a constant integer}}
}

void test_amdgcn_wmma_bf16f32_16x16x32_bf16(global v8bf16* out, v16bf16 a, v16bf16 b, v8f c, int mod)
{
  *out = __builtin_amdgcn_wmma_bf16f32_16x16x32_bf16(mod, a, 0, b, 0, c, false, false); // expected-error {{'__builtin_amdgcn_wmma_bf16f32_16x16x32_bf16' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_bf16f32_16x16x32_bf16(0, a, mod, b, 0, c, false, false); // expected-error {{'__builtin_amdgcn_wmma_bf16f32_16x16x32_bf16' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_bf16f32_16x16x32_bf16(0, a, 0, b, mod, c, false, false); // expected-error {{'__builtin_amdgcn_wmma_bf16f32_16x16x32_bf16' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_bf16f32_16x16x32_bf16(0, a, 0, b, 0, c, mod, false); // expected-error {{'__builtin_amdgcn_wmma_bf16f32_16x16x32_bf16' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_bf16f32_16x16x32_bf16(0, a, 0, b, 0, c, false, mod); // expected-error {{'__builtin_amdgcn_wmma_bf16f32_16x16x32_bf16' must be a constant integer}}
}

void test_amdgcn_wmma_f32_16x16x64_fp8_fp8(global v8f* out, v8i a, v8i b, v8f c, int mod)
{
  *out = __builtin_amdgcn_wmma_f32_16x16x64_fp8_fp8(a, b, mod, c, false, false); // expected-error {{'__builtin_amdgcn_wmma_f32_16x16x64_fp8_fp8' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_f32_16x16x64_fp8_fp8(a, b, 0, c, mod, false); // expected-error {{'__builtin_amdgcn_wmma_f32_16x16x64_fp8_fp8' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_f32_16x16x64_fp8_fp8(a, b, 0, c, false, mod); // expected-error {{'__builtin_amdgcn_wmma_f32_16x16x64_fp8_fp8' must be a constant integer}}
}

void test_amdgcn_wmma_f32_16x16x64_fp8_bf8(global v8f* out, v8i a, v8i b, v8f c, int mod)
{
  *out = __builtin_amdgcn_wmma_f32_16x16x64_fp8_bf8(a, b, mod, c, false, false); // expected-error {{'__builtin_amdgcn_wmma_f32_16x16x64_fp8_bf8' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_f32_16x16x64_fp8_bf8(a, b, 0, c, mod, false); // expected-error {{'__builtin_amdgcn_wmma_f32_16x16x64_fp8_bf8' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_f32_16x16x64_fp8_bf8(a, b, 0, c, false, mod); // expected-error {{'__builtin_amdgcn_wmma_f32_16x16x64_fp8_bf8' must be a constant integer}}
}

void test_amdgcn_wmma_f32_16x16x64_bf8_fp8(global v8f* out, v8i a, v8i b, v8f c, int mod)
{
  *out = __builtin_amdgcn_wmma_f32_16x16x64_bf8_fp8(a, b, mod, c, false, false); // expected-error {{'__builtin_amdgcn_wmma_f32_16x16x64_bf8_fp8' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_f32_16x16x64_bf8_fp8(a, b, 0, c, mod, false); // expected-error {{'__builtin_amdgcn_wmma_f32_16x16x64_bf8_fp8' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_f32_16x16x64_bf8_fp8(a, b, 0, c, false, mod); // expected-error {{'__builtin_amdgcn_wmma_f32_16x16x64_bf8_fp8' must be a constant integer}}
}

void test_amdgcn_wmma_f32_16x16x64_bf8_bf8(global v8f* out, v8i a, v8i b, v8f c, int mod)
{
  *out = __builtin_amdgcn_wmma_f32_16x16x64_bf8_bf8(a, b, mod, c, false, false); // expected-error {{'__builtin_amdgcn_wmma_f32_16x16x64_bf8_bf8' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_f32_16x16x64_bf8_bf8(a, b, 0, c, mod, false); // expected-error {{'__builtin_amdgcn_wmma_f32_16x16x64_bf8_bf8' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_f32_16x16x64_bf8_bf8(a, b, 0, c, false, mod); // expected-error {{'__builtin_amdgcn_wmma_f32_16x16x64_bf8_bf8' must be a constant integer}}
}

void test_amdgcn_wmma_f16_16x16x64_fp8_fp8(global v8h* out, v8i a, v8i b, v8h c, int mod)
{
  *out = __builtin_amdgcn_wmma_f16_16x16x64_fp8_fp8(a, b, mod, c, false, false); // expected-error {{'__builtin_amdgcn_wmma_f16_16x16x64_fp8_fp8' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_f16_16x16x64_fp8_fp8(a, b, 0, c, mod, false); // expected-error {{'__builtin_amdgcn_wmma_f16_16x16x64_fp8_fp8' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_f16_16x16x64_fp8_fp8(a, b, 0, c, false, mod); // expected-error {{'__builtin_amdgcn_wmma_f16_16x16x64_fp8_fp8' must be a constant integer}}
}

void test_amdgcn_wmma_f16_16x16x64_fp8_bf8(global v8h* out, v8i a, v8i b, v8h c, int mod)
{
  *out = __builtin_amdgcn_wmma_f16_16x16x64_fp8_bf8(a, b, mod, c, false, false); // expected-error {{'__builtin_amdgcn_wmma_f16_16x16x64_fp8_bf8' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_f16_16x16x64_fp8_bf8(a, b, 0, c, mod, false); // expected-error {{'__builtin_amdgcn_wmma_f16_16x16x64_fp8_bf8' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_f16_16x16x64_fp8_bf8(a, b, 0, c, false, mod); // expected-error {{'__builtin_amdgcn_wmma_f16_16x16x64_fp8_bf8' must be a constant integer}}
}

void test_amdgcn_wmma_f16_16x16x64_bf8_fp8(global v8h* out, v8i a, v8i b, v8h c, int mod)
{
  *out = __builtin_amdgcn_wmma_f16_16x16x64_bf8_fp8(a, b, mod, c, false, false); // expected-error {{'__builtin_amdgcn_wmma_f16_16x16x64_bf8_fp8' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_f16_16x16x64_bf8_fp8(a, b, 0, c, mod, false); // expected-error {{'__builtin_amdgcn_wmma_f16_16x16x64_bf8_fp8' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_f16_16x16x64_bf8_fp8(a, b, 0, c, false, mod); // expected-error {{'__builtin_amdgcn_wmma_f16_16x16x64_bf8_fp8' must be a constant integer}}
}

void test_amdgcn_wmma_f16_16x16x64_bf8_bf8(global v8h* out, v8i a, v8i b, v8h c, int mod)
{
  *out = __builtin_amdgcn_wmma_f16_16x16x64_bf8_bf8(a, b, mod, c, false, false); // expected-error {{'__builtin_amdgcn_wmma_f16_16x16x64_bf8_bf8' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_f16_16x16x64_bf8_bf8(a, b, 0, c, mod, false); // expected-error {{'__builtin_amdgcn_wmma_f16_16x16x64_bf8_bf8' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_f16_16x16x64_bf8_bf8(a, b, 0, c, false, mod); // expected-error {{'__builtin_amdgcn_wmma_f16_16x16x64_bf8_bf8' must be a constant integer}}
}

void test_amdgcn_wmma_i32_16x16x64_iu8(global v8i* out, v8i a, v8i b, v8i c, int mod)
{
  *out = __builtin_amdgcn_wmma_i32_16x16x64_iu8(mod, a, 0, b, c, false, false); // expected-error {{'__builtin_amdgcn_wmma_i32_16x16x64_iu8' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_i32_16x16x64_iu8(0, a, mod, b, c, false, false); // expected-error {{'__builtin_amdgcn_wmma_i32_16x16x64_iu8' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_i32_16x16x64_iu8(0, a, 0, b, c, mod, false); // expected-error {{'__builtin_amdgcn_wmma_i32_16x16x64_iu8' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_i32_16x16x64_iu8(0, a, 0, b, c, false, mod); // expected-error {{'__builtin_amdgcn_wmma_i32_16x16x64_iu8' must be a constant integer}}
}

void test_amdgcn_wmma_f32_16x16x128_f8f6f4(global v8f* out, v16i a, v16i b, v8f c, int mod)
{
  *out = __builtin_amdgcn_wmma_f32_16x16x128_f8f6f4(mod, a, 2, b, 0, c); // expected-error {{'__builtin_amdgcn_wmma_f32_16x16x128_f8f6f4' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_f32_16x16x128_f8f6f4(1, a, mod, b, 0, c); // expected-error {{'__builtin_amdgcn_wmma_f32_16x16x128_f8f6f4' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_f32_16x16x128_f8f6f4(1, a, 2, b, mod, c); // expected-error {{'__builtin_amdgcn_wmma_f32_16x16x128_f8f6f4' must be a constant integer}}
}

void test_amdgcn_wmma_f32_16x16x32_f16(global v8f* out, v16h a, v16h b, v8f c, int mod)
{
  *out = __builtin_amdgcn_wmma_f32_16x16x32_f16(mod, a, 0, b, 0, c, false, false); // expected-error {{'__builtin_amdgcn_wmma_f32_16x16x32_f16' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_f32_16x16x32_f16(0, a, mod, b, 0, c, false, false); // expected-error {{'__builtin_amdgcn_wmma_f32_16x16x32_f16' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_f32_16x16x32_f16(0, a, 0, b, mod, c, false, false); // expected-error {{'__builtin_amdgcn_wmma_f32_16x16x32_f16' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_f32_16x16x32_f16(0, a, 0, b, 0, c, mod, false); // expected-error {{'__builtin_amdgcn_wmma_f32_16x16x32_f16' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_f32_16x16x32_f16(0, a, 0, b, 0, c, false, mod); // expected-error {{'__builtin_amdgcn_wmma_f32_16x16x32_f16' must be a constant integer}}
}

void test_amdgcn_wmma_f16_16x16x32_f16(global v8h* out, v16h a, v16h b, v8h c, int mod)
{
  *out = __builtin_amdgcn_wmma_f16_16x16x32_f16(mod, a, 0, b, 0, c, false, false); // expected-error {{'__builtin_amdgcn_wmma_f16_16x16x32_f16' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_f16_16x16x32_f16(0, a, mod, b, 0, c, false, false); // expected-error {{'__builtin_amdgcn_wmma_f16_16x16x32_f16' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_f16_16x16x32_f16(0, a, 0, b, mod, c, false, false); // expected-error {{'__builtin_amdgcn_wmma_f16_16x16x32_f16' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_f16_16x16x32_f16(0, a, 0, b, 0, c, mod, false); // expected-error {{'__builtin_amdgcn_wmma_f16_16x16x32_f16' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_f16_16x16x32_f16(0, a, 0, b, 0, c, false, mod); // expected-error {{'__builtin_amdgcn_wmma_f16_16x16x32_f16' must be a constant integer}}
}

void test_amdgcn_wmma_f16_16x16x128_fp8_fp8(global v8h* out, v16i a, v16i b, v8h c, int mod)
{
  *out = __builtin_amdgcn_wmma_f16_16x16x128_fp8_fp8(a, b, mod, c, false, false); // expected-error {{'__builtin_amdgcn_wmma_f16_16x16x128_fp8_fp8' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_f16_16x16x128_fp8_fp8(a, b, 0, c, mod, false); // expected-error {{'__builtin_amdgcn_wmma_f16_16x16x128_fp8_fp8' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_f16_16x16x128_fp8_fp8(a, b, 0, c, false, mod); // expected-error {{'__builtin_amdgcn_wmma_f16_16x16x128_fp8_fp8' must be a constant integer}}
}

void test_amdgcn_wmma_f16_16x16x128_fp8_bf8(global v8h* out, v16i a, v16i b, v8h c, int mod)
{
  *out = __builtin_amdgcn_wmma_f16_16x16x128_fp8_bf8(a, b, mod, c, false, false); // expected-error {{'__builtin_amdgcn_wmma_f16_16x16x128_fp8_bf8' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_f16_16x16x128_fp8_bf8(a, b, 0, c, mod, false); // expected-error {{'__builtin_amdgcn_wmma_f16_16x16x128_fp8_bf8' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_f16_16x16x128_fp8_bf8(a, b, 0, c, false, mod); // expected-error {{'__builtin_amdgcn_wmma_f16_16x16x128_fp8_bf8' must be a constant integer}}
}

void test_amdgcn_wmma_f16_16x16x128_bf8_fp8(global v8h* out, v16i a, v16i b, v8h c, int mod)
{
  *out = __builtin_amdgcn_wmma_f16_16x16x128_bf8_fp8(a, b, mod, c, false, false); // expected-error {{'__builtin_amdgcn_wmma_f16_16x16x128_bf8_fp8' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_f16_16x16x128_bf8_fp8(a, b, 0, c, mod, false); // expected-error {{'__builtin_amdgcn_wmma_f16_16x16x128_bf8_fp8' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_f16_16x16x128_bf8_fp8(a, b, 0, c, false, mod); // expected-error {{'__builtin_amdgcn_wmma_f16_16x16x128_bf8_fp8' must be a constant integer}}
}

void test_amdgcn_wmma_f16_16x16x128_bf8_bf8(global v8h* out, v16i a, v16i b, v8h c, int mod)
{
  *out = __builtin_amdgcn_wmma_f16_16x16x128_bf8_bf8(a, b, mod, c, false, false); // expected-error {{'__builtin_amdgcn_wmma_f16_16x16x128_bf8_bf8' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_f16_16x16x128_bf8_bf8(a, b, 0, c, mod, false); // expected-error {{'__builtin_amdgcn_wmma_f16_16x16x128_bf8_bf8' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_f16_16x16x128_bf8_bf8(a, b, 0, c, false, mod); // expected-error {{'__builtin_amdgcn_wmma_f16_16x16x128_bf8_bf8' must be a constant integer}}
}

void test_amdgcn_wmma_f32_16x16x128_fp8_fp8(global v8f* out, v16i a, v16i b, v8f c, int mod)
{
  *out = __builtin_amdgcn_wmma_f32_16x16x128_fp8_fp8(a, b, mod, c, false, false); // expected-error {{'__builtin_amdgcn_wmma_f32_16x16x128_fp8_fp8' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_f32_16x16x128_fp8_fp8(a, b, 0, c, mod, false); // expected-error {{'__builtin_amdgcn_wmma_f32_16x16x128_fp8_fp8' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_f32_16x16x128_fp8_fp8(a, b, 0, c, false, mod); // expected-error {{'__builtin_amdgcn_wmma_f32_16x16x128_fp8_fp8' must be a constant integer}}
}

void test_amdgcn_wmma_f32_16x16x128_fp8_bf8(global v8f* out, v16i a, v16i b, v8f c, int mod)
{
  *out = __builtin_amdgcn_wmma_f32_16x16x128_fp8_bf8(a, b, mod, c, false, false); // expected-error {{'__builtin_amdgcn_wmma_f32_16x16x128_fp8_bf8' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_f32_16x16x128_fp8_bf8(a, b, 0, c, mod, false); // expected-error {{'__builtin_amdgcn_wmma_f32_16x16x128_fp8_bf8' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_f32_16x16x128_fp8_bf8(a, b, 0, c, false, mod); // expected-error {{'__builtin_amdgcn_wmma_f32_16x16x128_fp8_bf8' must be a constant integer}}
}

void test_amdgcn_wmma_f32_16x16x128_bf8_fp8(global v8f* out, v16i a, v16i b, v8f c, int mod)
{
  *out = __builtin_amdgcn_wmma_f32_16x16x128_bf8_fp8(a, b, mod, c, false, false); // expected-error {{'__builtin_amdgcn_wmma_f32_16x16x128_bf8_fp8' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_f32_16x16x128_bf8_fp8(a, b, 0, c, mod, false); // expected-error {{'__builtin_amdgcn_wmma_f32_16x16x128_bf8_fp8' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_f32_16x16x128_bf8_fp8(a, b, 0, c, false, mod); // expected-error {{'__builtin_amdgcn_wmma_f32_16x16x128_bf8_fp8' must be a constant integer}}
}

void test_amdgcn_wmma_f32_16x16x128_bf8_bf8(global v8f* out, v16i a, v16i b, v8f c, int mod)
{
  *out = __builtin_amdgcn_wmma_f32_16x16x128_bf8_bf8(a, b, mod, c, false, false); // expected-error {{'__builtin_amdgcn_wmma_f32_16x16x128_bf8_bf8' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_f32_16x16x128_bf8_bf8(a, b, 0, c, mod, false); // expected-error {{'__builtin_amdgcn_wmma_f32_16x16x128_bf8_bf8' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_f32_16x16x128_bf8_bf8(a, b, 0, c, false, mod); // expected-error {{'__builtin_amdgcn_wmma_f32_16x16x128_bf8_bf8' must be a constant integer}}
}

void test_amdgcn_wmma_f32_wmma_f32_32x16x128_f4(global v16f* out, v16i a, v8i b, v16f c, int mod)
{
  *out = __builtin_amdgcn_wmma_f32_32x16x128_f4(a, b, mod, c); // expected-error {{'__builtin_amdgcn_wmma_f32_32x16x128_f4' must be a constant integer}}
}

void test_amdgcn_swmmac_f32_16x16x64_bf16(global v8f* out, v16bf16 a, v32bf16 b, v8f c, int index, int mod)
{
  *out = __builtin_amdgcn_swmmac_f32_16x16x64_bf16(mod, a, 0, b, c, index, false, false); // expected-error {{'__builtin_amdgcn_swmmac_f32_16x16x64_bf16' must be a constant integer}}
  *out = __builtin_amdgcn_swmmac_f32_16x16x64_bf16(0, a, mod, b, c, index, false, false); // expected-error {{'__builtin_amdgcn_swmmac_f32_16x16x64_bf16' must be a constant integer}}
  *out = __builtin_amdgcn_swmmac_f32_16x16x64_bf16(0, a, 0, b, c, index, mod, false); // expected-error {{'__builtin_amdgcn_swmmac_f32_16x16x64_bf16' must be a constant integer}}
  *out = __builtin_amdgcn_swmmac_f32_16x16x64_bf16(0, a, 0, b, c, index, false, mod); // expected-error {{'__builtin_amdgcn_swmmac_f32_16x16x64_bf16' must be a constant integer}}
}

void test_amdgcn_swmmac_bf16_16x16x64_bf16(global v8bf16* out, v16bf16 a, v32bf16 b, v8bf16 c, int index, int mod)
{
  *out = __builtin_amdgcn_swmmac_bf16_16x16x64_bf16(mod, a, 0, b, c, index, false, false); // expected-error {{'__builtin_amdgcn_swmmac_bf16_16x16x64_bf16' must be a constant integer}}
  *out = __builtin_amdgcn_swmmac_bf16_16x16x64_bf16(0, a, mod, b, c, index, false, false); // expected-error {{'__builtin_amdgcn_swmmac_bf16_16x16x64_bf16' must be a constant integer}}
  *out = __builtin_amdgcn_swmmac_bf16_16x16x64_bf16(0, a, 0, b, c, index, mod, false); // expected-error {{'__builtin_amdgcn_swmmac_bf16_16x16x64_bf16' must be a constant integer}}
  *out = __builtin_amdgcn_swmmac_bf16_16x16x64_bf16(0, a, 0, b, c, index, false, mod); // expected-error {{'__builtin_amdgcn_swmmac_bf16_16x16x64_bf16' must be a constant integer}}
}

void test_amdgcn_swmmac_bf16f32_16x16x64_bf16(global v8f* out, v16bf16 a, v32bf16 b, v8f c, int index, int mod)
{
  *out = __builtin_amdgcn_swmmac_bf16f32_16x16x64_bf16(mod, a, 0, b, c, index, false, false); // expected-error {{'__builtin_amdgcn_swmmac_bf16f32_16x16x64_bf16' must be a constant integer}}
  *out = __builtin_amdgcn_swmmac_bf16f32_16x16x64_bf16(0, a, mod, b, c, index, false, false); // expected-error {{'__builtin_amdgcn_swmmac_bf16f32_16x16x64_bf16' must be a constant integer}}
  *out = __builtin_amdgcn_swmmac_bf16f32_16x16x64_bf16(0, a, 0, b, c, index, mod, false); // expected-error {{'__builtin_amdgcn_swmmac_bf16f32_16x16x64_bf16' must be a constant integer}}
  *out = __builtin_amdgcn_swmmac_bf16f32_16x16x64_bf16(0, a, 0, b, c, index, false, mod); // expected-error {{'__builtin_amdgcn_swmmac_bf16f32_16x16x64_bf16' must be a constant integer}}
}

void test_amdgcn_swmmac_i32_16x16x128_iu8(global v8i* out, v8i a, v16i b, v8i c, int index, int mod)
{
  *out = __builtin_amdgcn_swmmac_i32_16x16x128_iu8(mod, a, 0, b, c, index, false, false); // expected-error {{'__builtin_amdgcn_swmmac_i32_16x16x128_iu8' must be a constant integer}}
  *out = __builtin_amdgcn_swmmac_i32_16x16x128_iu8(0, a, mod, b, c, index, false, false); // expected-error {{'__builtin_amdgcn_swmmac_i32_16x16x128_iu8' must be a constant integer}}
  *out = __builtin_amdgcn_swmmac_i32_16x16x128_iu8(0, a, 0, b, c, index, mod, false); // expected-error {{'__builtin_amdgcn_swmmac_i32_16x16x128_iu8' must be a constant integer}}
  *out = __builtin_amdgcn_swmmac_i32_16x16x128_iu8(0, a, 0, b, c, index, false, mod); // expected-error {{'__builtin_amdgcn_swmmac_i32_16x16x128_iu8' must be a constant integer}}
}

void test_amdgcn_swmmac_f32_16x16x64_f16(global v8f* out, v16h a, v32h b, v8f c, int index, int mod)
{
  *out = __builtin_amdgcn_swmmac_f32_16x16x64_f16(mod, a, 0, b, c, index, false, false); // expected-error {{'__builtin_amdgcn_swmmac_f32_16x16x64_f16' must be a constant integer}}
  *out = __builtin_amdgcn_swmmac_f32_16x16x64_f16(0, a, mod, b, c, index, false, false); // expected-error {{'__builtin_amdgcn_swmmac_f32_16x16x64_f16' must be a constant integer}}
  *out = __builtin_amdgcn_swmmac_f32_16x16x64_f16(0, a, 0, b, c, index, mod, false); // expected-error {{'__builtin_amdgcn_swmmac_f32_16x16x64_f16' must be a constant integer}}
  *out = __builtin_amdgcn_swmmac_f32_16x16x64_f16(0, a, 0, b, c, index, false, mod); // expected-error {{'__builtin_amdgcn_swmmac_f32_16x16x64_f16' must be a constant integer}}
}

void test_amdgcn_swmmac_f16_16x16x64_f16(global v8h* out, v16h a, v32h b, v8h c, int index, int mod)
{
  *out = __builtin_amdgcn_swmmac_f16_16x16x64_f16(mod, a, 0, b, c, index, false, false); // expected-error {{'__builtin_amdgcn_swmmac_f16_16x16x64_f16' must be a constant integer}}
  *out = __builtin_amdgcn_swmmac_f16_16x16x64_f16(0, a, mod, b, c, index, false, false); // expected-error {{'__builtin_amdgcn_swmmac_f16_16x16x64_f16' must be a constant integer}}
  *out = __builtin_amdgcn_swmmac_f16_16x16x64_f16(0, a, 0, b, c, index, mod, false); // expected-error {{'__builtin_amdgcn_swmmac_f16_16x16x64_f16' must be a constant integer}}
  *out = __builtin_amdgcn_swmmac_f16_16x16x64_f16(0, a, 0, b, c, index, false, mod); // expected-error {{'__builtin_amdgcn_swmmac_f16_16x16x64_f16' must be a constant integer}}
}
