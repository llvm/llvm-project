// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1251 -verify -emit-llvm -o - %s

typedef double v8d   __attribute__((ext_vector_type(8)));
typedef double v4d   __attribute__((ext_vector_type(4)));
typedef double v2d   __attribute__((ext_vector_type(2)));
typedef float  v8f   __attribute__((ext_vector_type(8)));
typedef half   v8h   __attribute__((ext_vector_type(8)));

void test_amdgcn_wmma_f64_16x16x4_f64(global v8d* out, v2d a, v2d b, v8d c, int mod)
{
  *out = __builtin_amdgcn_wmma_f64_16x16x4_f64(mod, a, 0, b, 0, c); // expected-error {{'__builtin_amdgcn_wmma_f64_16x16x4_f64' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_f64_16x16x4_f64(0, a, mod, b, 0, c); // expected-error {{'__builtin_amdgcn_wmma_f64_16x16x4_f64' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_f64_16x16x4_f64(0, a, 0, b, mod, c); // expected-error {{'__builtin_amdgcn_wmma_f64_16x16x4_f64' must be a constant integer}}
}

void test_amdgcn_wmma_f64_16x16x8_f64(global v8d* out, v4d a, v4d b, v8d c, int mod)
{
  *out = __builtin_amdgcn_wmma_f64_16x16x8_f64(mod, a, 0, b, 0, c); // expected-error {{'__builtin_amdgcn_wmma_f64_16x16x8_f64' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_f64_16x16x8_f64(0, a, mod, b, 0, c); // expected-error {{'__builtin_amdgcn_wmma_f64_16x16x8_f64' must be a constant integer}}
  *out = __builtin_amdgcn_wmma_f64_16x16x8_f64(0, a, 0, b, mod, c); // expected-error {{'__builtin_amdgcn_wmma_f64_16x16x8_f64' must be a constant integer}}
}
