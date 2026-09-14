// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgpu12.50s-- -verify -S -o - %s

typedef float  v16f  __attribute__((ext_vector_type(16)));
typedef float  v8f   __attribute__((ext_vector_type(8)));
typedef int    v16i   __attribute__((ext_vector_type(16)));
typedef int    v8i   __attribute__((ext_vector_type(8)));

void test_amdgcn_wmma_f32_wmma_f4(global v16f* out, v16i a, v8i b, v16f c, int scale_src0, int scale_src1)
{
  *out = __builtin_amdgcn_wmma_f32_32x16x128_f4(a, b, false, c); // expected-error {{'__builtin_amdgcn_wmma_f32_32x16x128_f4' needs target feature wmma-f4-insts}}
  *out = __builtin_amdgcn_wmma_scale_f32_32x16x128_f4(a, b, false, c, 1, 0, scale_src0, 2, 0, scale_src1, 1, 0); // expected-error {{'__builtin_amdgcn_wmma_scale_f32_32x16x128_f4' needs target feature wmma-f4-insts}}
  *out = __builtin_amdgcn_wmma_scale16_f32_32x16x128_f4(a, b, false, c, 1, 0, scale_src0, 2, 0, scale_src1, 1, 0); // expected-error {{'__builtin_amdgcn_wmma_scale16_f32_32x16x128_f4' needs target feature wmma-f4-insts}}
}
