// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -cl-std=CL2.0 -triple amdgpu13.10-unknown-unknown \
// RUN:   -target-feature +wavefrontsize64 -verify -S -o - %s

typedef unsigned int uint;
typedef unsigned int __attribute__((ext_vector_type(6))) uint6;
typedef __bf16 __attribute__((ext_vector_type(32))) bfloat32;
typedef half __attribute__((ext_vector_type(32))) half32;
typedef float __attribute__((ext_vector_type(32))) float32;

void test_cvt_scale_pk32_w64(global bfloat32 *outbf32, global half32 *outhalf32, global float32 *outf32, uint6 src, uint scale)
{
  *outbf32 = __builtin_amdgcn_cvt_scale_pk32_bf16_bf6(src, scale, 0); // expected-error {{'__builtin_amdgcn_cvt_scale_pk32_bf16_bf6' needs target feature fp6bf6-to-f16bf16f32-cvt-scale-insts,wavefrontsize32}}
  *outbf32 = __builtin_amdgcn_cvt_scale_pk32_bf16_fp6(src, scale, 0); // expected-error {{'__builtin_amdgcn_cvt_scale_pk32_bf16_fp6' needs target feature fp6bf6-to-f16bf16f32-cvt-scale-insts,wavefrontsize32}}
  *outhalf32 = __builtin_amdgcn_cvt_scale_pk32_f16_bf6(src, scale, 0); // expected-error {{'__builtin_amdgcn_cvt_scale_pk32_f16_bf6' needs target feature fp6bf6-to-f16bf16f32-cvt-scale-insts,wavefrontsize32}}
  *outhalf32 = __builtin_amdgcn_cvt_scale_pk32_f16_fp6(src, scale, 0); // expected-error {{'__builtin_amdgcn_cvt_scale_pk32_f16_fp6' needs target feature fp6bf6-to-f16bf16f32-cvt-scale-insts,wavefrontsize32}}
  *outf32 = __builtin_amdgcn_cvt_scale_pk32_f32_bf6(src, scale, 0); // expected-error {{'__builtin_amdgcn_cvt_scale_pk32_f32_bf6' needs target feature fp6bf6-to-f16bf16f32-cvt-scale-insts,wavefrontsize32}}
  *outf32 = __builtin_amdgcn_cvt_scale_pk32_f32_fp6(src, scale, 0); // expected-error {{'__builtin_amdgcn_cvt_scale_pk32_f32_fp6' needs target feature fp6bf6-to-f16bf16f32-cvt-scale-insts,wavefrontsize32}}
}
