// RUN: %clang_cc1 -triple amdgcn-- -target-cpu gfx1100 -target-feature +extended-image-insts -S -verify=expected -o - %s
// REQUIRES: amdgpu-registered-target

typedef int int4 __attribute__((ext_vector_type(4)));
typedef float float4 __attribute__((ext_vector_type(4)));
typedef half half4 __attribute__((ext_vector_type(4)));

float4 test_amdgcn_image_gather4_lz_2d_v4f32_f32_r(float4 v4f32, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_gather4_lz_2d_v4f32_f32(1, f32, f32, tex, vec4i32, 0, f32, 110); //expected-error{{argument to '__builtin_amdgcn_image_gather4_lz_2d_v4f32_f32' must be a constant integer}}
}

half4 test_amdgcn_image_sample_l_3d_v4f16_f32(float4 v4f32, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_l_3d_v4f16_f32(100, f32, f32, f32, f32, tex, vec4i32, 0, 120, i32); //expected-error{{argument to '__builtin_amdgcn_image_sample_l_3d_v4f16_f32' must be a constant integer}}
}

float test_amdgcn_image_sample_d_2d_f32_f32(float4 v4f32, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_d_2d_f32_f32(f32, f32, f32, f32, f32, f32, f32, tex, vec4i32, 0, 120, 110); //expected-error{{argument to '__builtin_amdgcn_image_sample_d_2d_f32_f32' must be a constant integer}}
}
