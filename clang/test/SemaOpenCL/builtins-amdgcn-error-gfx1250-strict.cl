// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgpu12.50s-- -verify -S -o - %s

typedef unsigned int uint;
typedef unsigned int __attribute__((ext_vector_type(2))) uint2;
typedef unsigned int __attribute__((ext_vector_type(3))) uint3;
typedef float  v16f  __attribute__((ext_vector_type(16)));
typedef float  v8f   __attribute__((ext_vector_type(8)));
typedef int    v16i   __attribute__((ext_vector_type(16)));
typedef int    v8i   __attribute__((ext_vector_type(8)));

void test(global v16f* out, v16i a, v8i b, v16f c, int scale_src0, int scale_src1, uint scale, uint src1, uint2 src2, uint3 src3)
{
  *out = __builtin_amdgcn_wmma_f32_32x16x128_f4(a, b, false, c); // expected-error {{'__builtin_amdgcn_wmma_f32_32x16x128_f4' needs target feature wmma-f4-insts}}
  *out = __builtin_amdgcn_wmma_scale_f32_32x16x128_f4(a, b, false, c, 1, 0, scale_src0, 2, 0, scale_src1, 1, 0); // expected-error {{'__builtin_amdgcn_wmma_scale_f32_32x16x128_f4' needs target feature wmma-f4-insts}}
  *out = __builtin_amdgcn_wmma_scale16_f32_32x16x128_f4(a, b, false, c, 1, 0, scale_src0, 2, 0, scale_src1, 1, 0); // expected-error {{'__builtin_amdgcn_wmma_scale16_f32_32x16x128_f4' needs target feature wmma-f4-insts}}
  (void)__builtin_amdgcn_cvt_scale_pk8_f16_fp8(src2, scale, 0); // expected-error {{'__builtin_amdgcn_cvt_scale_pk8_f16_fp8' needs target feature block16-cvt-scale-insts}}
  (void)__builtin_amdgcn_cvt_scale_pk8_bf16_fp8(src2, scale, 0); // expected-error {{'__builtin_amdgcn_cvt_scale_pk8_bf16_fp8' needs target feature block16-cvt-scale-insts}}
  (void)__builtin_amdgcn_cvt_scale_pk8_f16_bf8(src2, scale, 0); // expected-error {{'__builtin_amdgcn_cvt_scale_pk8_f16_bf8' needs target feature block16-cvt-scale-insts}}
  (void)__builtin_amdgcn_cvt_scale_pk8_bf16_bf8(src2, scale, 0); // expected-error {{'__builtin_amdgcn_cvt_scale_pk8_bf16_bf8' needs target feature block16-cvt-scale-insts}}
  (void)__builtin_amdgcn_cvt_scale_pk8_f16_fp4(src1, scale, 0); // expected-error {{'__builtin_amdgcn_cvt_scale_pk8_f16_fp4' needs target feature block16-cvt-scale-insts}}
  (void)__builtin_amdgcn_cvt_scale_pk8_bf16_fp4(src1, scale, 0); // expected-error {{'__builtin_amdgcn_cvt_scale_pk8_bf16_fp4' needs target feature block16-cvt-scale-insts}}
  (void)__builtin_amdgcn_cvt_scale_pk8_f32_fp8(src2, scale, 0); // expected-error {{'__builtin_amdgcn_cvt_scale_pk8_f32_fp8' needs target feature block16-cvt-scale-insts}}
  (void)__builtin_amdgcn_cvt_scale_pk8_f32_bf8(src2, scale, 0); // expected-error {{'__builtin_amdgcn_cvt_scale_pk8_f32_bf8' needs target feature block16-cvt-scale-insts}}
  (void)__builtin_amdgcn_cvt_scale_pk8_f32_fp4(src1, scale, 0); // expected-error {{'__builtin_amdgcn_cvt_scale_pk8_f32_fp4' needs target feature block16-cvt-scale-insts}}
  (void)__builtin_amdgcn_cvt_scale_pk16_f16_fp6(src3, scale, 0); // expected-error {{'__builtin_amdgcn_cvt_scale_pk16_f16_fp6' needs target feature block16-cvt-scale-insts}}
  (void)__builtin_amdgcn_cvt_scale_pk16_bf16_fp6(src3, scale, 0); // expected-error {{'__builtin_amdgcn_cvt_scale_pk16_bf16_fp6' needs target feature block16-cvt-scale-insts}}
  (void)__builtin_amdgcn_cvt_scale_pk16_f16_bf6(src3, scale, 0); // expected-error {{'__builtin_amdgcn_cvt_scale_pk16_f16_bf6' needs target feature block16-cvt-scale-insts}}
  (void)__builtin_amdgcn_cvt_scale_pk16_bf16_bf6(src3, scale, 0); // expected-error {{'__builtin_amdgcn_cvt_scale_pk16_bf16_bf6' needs target feature block16-cvt-scale-insts}}
  (void)__builtin_amdgcn_cvt_scale_pk16_f32_fp6(src3, scale, 0); // expected-error {{'__builtin_amdgcn_cvt_scale_pk16_f32_fp6' needs target feature block16-cvt-scale-insts}}
  (void)__builtin_amdgcn_cvt_scale_pk16_f32_bf6(src3, scale, 0); // expected-error {{'__builtin_amdgcn_cvt_scale_pk16_f32_bf6' needs target feature block16-cvt-scale-insts}}
}
