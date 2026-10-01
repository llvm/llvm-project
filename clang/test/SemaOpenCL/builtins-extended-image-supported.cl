// RUN: %clang_cc1 -triple amdgpu6.00-- -verify -fsyntax-only %s
// RUN: %clang_cc1 -triple amdgpu7.00-- -verify -fsyntax-only %s
// RUN: %clang_cc1 -triple amdgpu8.03-- -verify -fsyntax-only %s
// RUN: %clang_cc1 -triple amdgpu9-- -verify -fsyntax-only %s
// RUN: %clang_cc1 -triple amdgpu9.00-- -verify -fsyntax-only %s
// RUN: %clang_cc1 -triple amdgpu10.10-- -verify -fsyntax-only %s
// RUN: %clang_cc1 -triple amdgpu10.30-- -verify -fsyntax-only %s
// RUN: %clang_cc1 -triple amdgpu11-- -verify -fsyntax-only %s
// RUN: %clang_cc1 -triple amdgpu11.00-- -verify -fsyntax-only %s
// RUN: %clang_cc1 -triple amdgpu12.00-- -verify -fsyntax-only %s
// RUN: %clang_cc1 -triple amdgpu12-- -verify -fsyntax-only %s
// RUN: %clang_cc1 -triple amdgpu13.10-- -verify -fsyntax-only %s
// REQUIRES: amdgpu-registered-target

// expected-no-diagnostics

typedef int int4 __attribute__((ext_vector_type(4)));
typedef float float4 __attribute__((ext_vector_type(4)));

float4 test_image_sample_lz(float f32, __amdgpu_texture_t tex, int4 vec4i32) {
  return __builtin_amdgcn_image_sample_lz_2d_v4f32_f32(1, f32, f32, tex, vec4i32, 0, 0, 0);
}

float4 test_image_gather4_lz(float f32, __amdgpu_texture_t tex, int4 vec4i32) {
  return __builtin_amdgcn_image_gather4_lz_2d_v4f32_f32(1, f32, f32, tex, vec4i32, 0, 0, 0);
}
