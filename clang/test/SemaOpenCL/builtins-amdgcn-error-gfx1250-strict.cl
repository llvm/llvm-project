// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgpu12.50s-- -verify -S -o - %s

typedef unsigned int uint;
typedef unsigned int __attribute__((ext_vector_type(2))) uint2;
typedef unsigned int __attribute__((ext_vector_type(3))) uint3;
typedef float  v16f  __attribute__((ext_vector_type(16)));
typedef float  v8f   __attribute__((ext_vector_type(8)));
typedef half   v8h   __attribute__((ext_vector_type(8)));
typedef int    v16i   __attribute__((ext_vector_type(16)));
typedef int    v8i   __attribute__((ext_vector_type(8)));

void test(global v16f* out, v16i a, v8i b, v16f c, int scale_src0, int scale_src1, uint scale, uint src1, uint2 src2, uint3 src3,
          global v8h* out8h, global v8f* out8f, v16i a16i, v16i b16i, v8h c8h, v8f c8f)
{
  *out = __builtin_amdgcn_wmma_f32_32x16x128_f4(a, b, false, c); // expected-error {{'__builtin_amdgcn_wmma_f32_32x16x128_f4' needs target feature wmma-f4-insts}}
  *out = __builtin_amdgcn_wmma_scale_f32_32x16x128_f4(a, b, false, c, 1, 0, scale_src0, 2, 0, scale_src1, 1, 0); // expected-error {{'__builtin_amdgcn_wmma_scale_f32_32x16x128_f4' needs target feature wmma-f4-insts}}
  *out = __builtin_amdgcn_wmma_scale16_f32_32x16x128_f4(a, b, false, c, 1, 0, scale_src0, 2, 0, scale_src1, 1, 0); // expected-error {{'__builtin_amdgcn_wmma_scale16_f32_32x16x128_f4' needs target feature wmma-f4-insts}}
  *out8h = __builtin_amdgcn_wmma_f16_16x16x128_fp8_fp8(a16i, b16i, 0, c8h, false, true); // expected-error {{'__builtin_amdgcn_wmma_f16_16x16x128_fp8_fp8' needs target feature wmma-16x16x128fp8-insts}}
  *out8h = __builtin_amdgcn_wmma_f16_16x16x128_fp8_bf8(a16i, b16i, 0, c8h, false, true); // expected-error {{'__builtin_amdgcn_wmma_f16_16x16x128_fp8_bf8' needs target feature wmma-16x16x128fp8-insts}}
  *out8h = __builtin_amdgcn_wmma_f16_16x16x128_bf8_fp8(a16i, b16i, 0, c8h, false, true); // expected-error {{'__builtin_amdgcn_wmma_f16_16x16x128_bf8_fp8' needs target feature wmma-16x16x128fp8-insts}}
  *out8h = __builtin_amdgcn_wmma_f16_16x16x128_bf8_bf8(a16i, b16i, 0, c8h, false, true); // expected-error {{'__builtin_amdgcn_wmma_f16_16x16x128_bf8_bf8' needs target feature wmma-16x16x128fp8-insts}}
  *out8f = __builtin_amdgcn_wmma_f32_16x16x128_fp8_fp8(a16i, b16i, 0, c8f, false, true); // expected-error {{'__builtin_amdgcn_wmma_f32_16x16x128_fp8_fp8' needs target feature wmma-16x16x128fp8-insts}}
  *out8f = __builtin_amdgcn_wmma_f32_16x16x128_fp8_bf8(a16i, b16i, 0, c8f, false, true); // expected-error {{'__builtin_amdgcn_wmma_f32_16x16x128_fp8_bf8' needs target feature wmma-16x16x128fp8-insts}}
  *out8f = __builtin_amdgcn_wmma_f32_16x16x128_bf8_fp8(a16i, b16i, 0, c8f, false, true); // expected-error {{'__builtin_amdgcn_wmma_f32_16x16x128_bf8_fp8' needs target feature wmma-16x16x128fp8-insts}}
  *out8f = __builtin_amdgcn_wmma_f32_16x16x128_bf8_bf8(a16i, b16i, 0, c8f, false, true); // expected-error {{'__builtin_amdgcn_wmma_f32_16x16x128_bf8_bf8' needs target feature wmma-16x16x128fp8-insts}}
}
