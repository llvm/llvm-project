// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -cl-std=CL2.0 -triple amdgpu13.10-unknown-unknown -verify -emit-llvm -o - %s

typedef unsigned int uint;
typedef unsigned int __attribute__((ext_vector_type(6))) uint6;
typedef __bf16 __attribute__((ext_vector_type(32))) bfloat32;
typedef half __attribute__((ext_vector_type(32))) half32;
typedef float __attribute__((ext_vector_type(32))) float32;

void test_cvt_scale_pk32(global bfloat32 *outbf32, global half32 *outhalf32, global float32 *outf32, uint6 src, uint scale, uint scale_sel)
{
  *outbf32 = __builtin_amdgcn_cvt_scale_pk32_bf16_bf6(src, scale, scale_sel); // expected-error {{'__builtin_amdgcn_cvt_scale_pk32_bf16_bf6' must be a constant integer}}
  *outbf32 = __builtin_amdgcn_cvt_scale_pk32_bf16_fp6(src, scale, scale_sel); // expected-error {{'__builtin_amdgcn_cvt_scale_pk32_bf16_fp6' must be a constant integer}}
  *outhalf32 = __builtin_amdgcn_cvt_scale_pk32_f16_bf6(src, scale, scale_sel); // expected-error {{'__builtin_amdgcn_cvt_scale_pk32_f16_bf6' must be a constant integer}}
  *outhalf32 = __builtin_amdgcn_cvt_scale_pk32_f16_fp6(src, scale, scale_sel); // expected-error {{'__builtin_amdgcn_cvt_scale_pk32_f16_fp6' must be a constant integer}}
  *outf32 = __builtin_amdgcn_cvt_scale_pk32_f32_bf6(src, scale, scale_sel); // expected-error {{'__builtin_amdgcn_cvt_scale_pk32_f32_bf6' must be a constant integer}}
  *outf32 = __builtin_amdgcn_cvt_scale_pk32_f32_fp6(src, scale, scale_sel); // expected-error {{'__builtin_amdgcn_cvt_scale_pk32_f32_fp6' must be a constant integer}}

  *outbf32 = __builtin_amdgcn_cvt_scale_pk32_bf16_bf6(src, scale, 16);  // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  *outbf32 = __builtin_amdgcn_cvt_scale_pk32_bf16_fp6(src, scale, 16);  // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  *outhalf32 = __builtin_amdgcn_cvt_scale_pk32_f16_bf6(src, scale, 16);  // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  *outhalf32 = __builtin_amdgcn_cvt_scale_pk32_f16_fp6(src, scale, 16);  // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  *outf32 = __builtin_amdgcn_cvt_scale_pk32_f32_bf6(src, scale, 16);  // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  *outf32 = __builtin_amdgcn_cvt_scale_pk32_f32_fp6(src, scale, 16);  // expected-error {{argument value 16 is outside the valid range [0, 15]}}
}
