// RUN: %clang_cc1 -triple amdgcn-- -target-cpu gfx942 -verify=GFX94 -S -o - %s
// REQUIRES: amdgpu-registered-target

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

typedef int int4 __attribute__((ext_vector_type(4)));
typedef float float4 __attribute__((ext_vector_type(4)));
typedef half half4 __attribute__((ext_vector_type(4)));

float4 test_amdgcn_image_gather4_lz_2d_v4f32_f32_r(float4 v4f32, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_gather4_lz_2d_v4f32_f32(1, f32, f32, tex, vec4i32, 0, 101, 121); //GFX94-error{{'test_amdgcn_image_gather4_lz_2d_v4f32_f32_r' needs target feature extended-image-insts}}
}

float4 test_amdgcn_image_gather4_lz_2d_v4f32_f32_g(float4 v4f32, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_gather4_lz_2d_v4f32_f32(2, f32, f32, tex, vec4i32, 0, 101, 121); //GFX94-error{{'test_amdgcn_image_gather4_lz_2d_v4f32_f32_g' needs target feature extended-image-insts}}
}

float4 test_amdgcn_image_gather4_lz_2d_v4f32_f32_b(float4 v4f32, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_gather4_lz_2d_v4f32_f32(4, f32, f32, tex, vec4i32, 0, 101, 121); //GFX94-error{{'test_amdgcn_image_gather4_lz_2d_v4f32_f32_b' needs target feature extended-image-insts}}
}

float4 test_amdgcn_image_gather4_lz_2d_v4f32_f32_a(float4 v4f32, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_gather4_lz_2d_v4f32_f32(8, f32, f32, tex, vec4i32, 0, 101, 121); //GFX94-error{{'test_amdgcn_image_gather4_lz_2d_v4f32_f32_a' needs target feature extended-image-insts}}
}

float4 test_amdgcn_image_sample_lz_1d_v4f32_f32(float4 v4f32, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_lz_1d_v4f32_f32(105, f32, tex, vec4i32, 0, 101, 121); //GFX94-error{{'test_amdgcn_image_sample_lz_1d_v4f32_f32' needs target feature extended-image-insts}}
}

float4 test_amdgcn_image_sample_l_1d_v4f32_f32(float4 v4f32, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_l_1d_v4f32_f32(100, f32, f32, tex, vec4i32, 0, 101, 121); //GFX94-error{{'test_amdgcn_image_sample_l_1d_v4f32_f32' needs target feature extended-image-insts}}
}

float4 test_amdgcn_image_sample_d_1d_v4f32_f32(float4 v4f32, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_d_1d_v4f32_f32(100, f32, f32, f32, tex, vec4i32, 0, 101, 121); //GFX94-error{{'test_amdgcn_image_sample_d_1d_v4f32_f32' needs target feature extended-image-insts}}
}

float4 test_amdgcn_image_sample_lz_2d_v4f32_f32(float4 v4f32, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_lz_2d_v4f32_f32(100, f32, f32, tex, vec4i32, 0, 101, 121); //GFX94-error{{'test_amdgcn_image_sample_lz_2d_v4f32_f32' needs target feature extended-image-insts}}
}

float4 test_amdgcn_image_sample_l_2d_v4f32_f32(float4 v4f32, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_l_2d_v4f32_f32(10, f32, f32, f32, tex, vec4i32, 0, 101, 121); //GFX94-error{{'test_amdgcn_image_sample_l_2d_v4f32_f32' needs target feature extended-image-insts}}
}

float4 test_amdgcn_image_sample_d_2d_v4f32_f32(float4 v4f32, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_d_2d_v4f32_f32(105, f32, f32, f32, f32, f32, f32, tex, vec4i32, 0, 101, 121); //GFX94-error{{'test_amdgcn_image_sample_d_2d_v4f32_f32' needs target feature extended-image-insts}}
}
float4 test_amdgcn_image_sample_lz_3d_v4f32_f32(float4 v4f32, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_lz_3d_v4f32_f32(105, f32, f32, f32, tex, vec4i32, 0, 101, 121); //GFX94-error{{'test_amdgcn_image_sample_lz_3d_v4f32_f32' needs target feature extended-image-insts}}
}

float4 test_amdgcn_image_sample_l_3d_v4f32_f32(float4 v4f32, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_l_3d_v4f32_f32(1, f32, f32, f32, f32, tex, vec4i32, 0, 101, 121); //GFX94-error{{'test_amdgcn_image_sample_l_3d_v4f32_f32' needs target feature extended-image-insts}}
}

float4 test_amdgcn_image_sample_d_3d_v4f32_f32(float4 v4f32, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_d_3d_v4f32_f32(1, f32, f32, f32, f32, f32, f32, f32, f32, f32, tex, vec4i32, 0, 101, 121); //GFX94-error{{'test_amdgcn_image_sample_d_3d_v4f32_f32' needs target feature extended-image-insts}}
}

float4 test_amdgcn_image_sample_lz_cube_v4f32_f32(float4 v4f32, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_lz_cube_v4f32_f32(1, f32, f32, f32, tex, vec4i32, 0, 101, 121); //GFX94-error{{'test_amdgcn_image_sample_lz_cube_v4f32_f32' needs target feature extended-image-insts}}
}

float4 test_amdgcn_image_sample_l_cube_v4f32_f32(float4 v4f32, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_l_cube_v4f32_f32(1, f32, f32, f32, f32, tex, vec4i32, 0, 101, 121); //GFX94-error{{'test_amdgcn_image_sample_l_cube_v4f32_f32' needs target feature extended-image-insts}}
}

float4 test_amdgcn_image_sample_lz_1darray_v4f32_f32(float4 v4f32, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_lz_1darray_v4f32_f32(1, f32, f32, tex, vec4i32, 0, 101, 121); //GFX94-error{{'test_amdgcn_image_sample_lz_1darray_v4f32_f32' needs target feature extended-image-insts}}
}

float4 test_amdgcn_image_sample_l_1darray_v4f32_f32(float4 v4f32, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_l_1darray_v4f32_f32(1, f32, f32, f32, tex, vec4i32, 0, 101, 121); //GFX94-error{{'test_amdgcn_image_sample_l_1darray_v4f32_f32' needs target feature extended-image-insts}}
}

float4 test_amdgcn_image_sample_d_1darray_v4f32_f32(float4 v4f32, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_d_1darray_v4f32_f32(1, f32, f32, f32, f32, tex, vec4i32, 0, 101, 121); //GFX94-error{{'test_amdgcn_image_sample_d_1darray_v4f32_f32' needs target feature extended-image-insts}}
}

float4 test_amdgcn_image_sample_lz_2darray_v4f32_f32(float4 v4f32, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_lz_2darray_v4f32_f32(1, f32, f32, f32, tex, vec4i32, 0, 101, 121); //GFX94-error{{'test_amdgcn_image_sample_lz_2darray_v4f32_f32' needs target feature extended-image-insts}}
}

float4 test_amdgcn_image_sample_l_2darray_v4f32_f32(float4 v4f32, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_l_2darray_v4f32_f32(1, f32, f32, f32, f32, tex, vec4i32, 0, 101, 121); //GFX94-error{{'test_amdgcn_image_sample_l_2darray_v4f32_f32' needs target feature extended-image-insts}}
}

float4 test_amdgcn_image_sample_d_2darray_v4f32_f32(float4 v4f32, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_d_2darray_v4f32_f32(1, f32, f32, f32, f32, f32, f32, f32, tex, vec4i32, 0, 101, 121); //GFX94-error{{'test_amdgcn_image_sample_d_2darray_v4f32_f32' needs target feature extended-image-insts}}
}

half4 test_amdgcn_image_sample_lz_1d_v4f16_f32(float4 v4f32, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_lz_1d_v4f16_f32(105, f32, tex, vec4i32, 0, 101, 121); //GFX94-error{{'test_amdgcn_image_sample_lz_1d_v4f16_f32' needs target feature extended-image-insts}}
}

half4 test_amdgcn_image_sample_l_1d_v4f16_f32(float4 v4f32, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_l_1d_v4f16_f32(105, f32, f32, tex, vec4i32, 0, 101, 121); //GFX94-error{{'test_amdgcn_image_sample_l_1d_v4f16_f32' needs target feature extended-image-insts}}
}

half4 test_amdgcn_image_sample_d_1d_v4f16_f32(float4 v4f32, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_d_1d_v4f16_f32(105, f32, f32, f32, tex, vec4i32, 0, 101, 121); //GFX94-error{{'test_amdgcn_image_sample_d_1d_v4f16_f32' needs target feature extended-image-insts}}
}

half4 test_amdgcn_image_sample_lz_2d_v4f16_f32(float4 v4f32, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_lz_2d_v4f16_f32(100, f32, f32, tex, vec4i32, 0, 101, 121); //GFX94-error{{'test_amdgcn_image_sample_lz_2d_v4f16_f32' needs target feature extended-image-insts}}
}

half4 test_amdgcn_image_sample_l_2d_v4f16_f32(float4 v4f32, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_l_2d_v4f16_f32(100, f32, f32, f32, tex, vec4i32, 0, 101, 121); //GFX94-error{{'test_amdgcn_image_sample_l_2d_v4f16_f32' needs target feature extended-image-insts}}
}

half4 test_amdgcn_image_sample_d_2d_v4f16_f32(float4 v4f32, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_d_2d_v4f16_f32(100, f32, f32, f32, f32, f32, f32, tex, vec4i32, 0, 101, 121); //GFX94-error{{'test_amdgcn_image_sample_d_2d_v4f16_f32' needs target feature extended-image-insts}}
}

half4 test_amdgcn_image_sample_lz_3d_v4f16_f32(float4 v4f32, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_lz_3d_v4f16_f32(100, f32, f32, f32, tex, vec4i32, 0, 101, 121); //GFX94-error{{'test_amdgcn_image_sample_lz_3d_v4f16_f32' needs target feature extended-image-insts}}
}

half4 test_amdgcn_image_sample_l_3d_v4f16_f32(float4 v4f32, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_l_3d_v4f16_f32(100, f32, f32, f32, f32, tex, vec4i32, 0, 101, 121); //GFX94-error{{'test_amdgcn_image_sample_l_3d_v4f16_f32' needs target feature extended-image-insts}}
}

half4 test_amdgcn_image_sample_d_3d_v4f16_f32(float4 v4f32, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_d_3d_v4f16_f32(100, f32, f32, f32, f32, f32, f32, f32, f32, f32, tex, vec4i32, 0, 101, 121); //GFX94-error{{'test_amdgcn_image_sample_d_3d_v4f16_f32' needs target feature extended-image-insts}}
}

half4 test_amdgcn_image_sample_lz_cube_v4f16_f32(float4 v4f32, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_lz_cube_v4f16_f32(100, f32, f32, f32, tex, vec4i32, 0, 101, 121); //GFX94-error{{'test_amdgcn_image_sample_lz_cube_v4f16_f32' needs target feature extended-image-insts}}
}

half4 test_amdgcn_image_sample_l_cube_v4f16_f32(float4 v4f32, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_l_cube_v4f16_f32(105, f32, f32, f32, f32, tex, vec4i32, 0, 101, 121); //GFX94-error{{'test_amdgcn_image_sample_l_cube_v4f16_f32' needs target feature extended-image-insts}}
}

half4 test_amdgcn_image_sample_lz_1darray_v4f16_f32(float4 v4f32, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_lz_1darray_v4f16_f32(105, f32, f32, tex, vec4i32, 0, 101, 121); //GFX94-error{{'test_amdgcn_image_sample_lz_1darray_v4f16_f32' needs target feature extended-image-insts}}
}

half4 test_amdgcn_image_sample_l_1darray_v4f16_f32(float4 v4f32, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_l_1darray_v4f16_f32(105, f32, f32, f32, tex, vec4i32, 0, 101, 121); //GFX94-error{{'test_amdgcn_image_sample_l_1darray_v4f16_f32' needs target feature extended-image-insts}}
}

half4 test_amdgcn_image_sample_d_1darray_v4f16_f32(float4 v4f32, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_d_1darray_v4f16_f32(100, f32, f32, f32, f32, tex, vec4i32, 0, 101, 121); //GFX94-error{{'test_amdgcn_image_sample_d_1darray_v4f16_f32' needs target feature extended-image-insts}}
}

half4 test_amdgcn_image_sample_lz_2darray_v4f16_f32(float4 v4f32, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_lz_2darray_v4f16_f32(100, f32, f32, f32, tex, vec4i32, 0, 101, 121); //GFX94-error{{'test_amdgcn_image_sample_lz_2darray_v4f16_f32' needs target feature extended-image-insts}}
}

half4 test_amdgcn_image_sample_l_2darray_v4f16_f32(float4 v4f32, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_l_2darray_v4f16_f32(100, f32, f32, f32, f32, tex, vec4i32, 0, 101, 121); //GFX94-error{{'test_amdgcn_image_sample_l_2darray_v4f16_f32' needs target feature extended-image-insts}}
}

half4 test_amdgcn_image_sample_d_2darray_v4f16_f32(float4 v4f32, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_d_2darray_v4f16_f32(100, f32, f32, f32, f32, f32, f32, f32, tex, vec4i32, 0, 101, 121); //GFX94-error{{'test_amdgcn_image_sample_d_2darray_v4f16_f32' needs target feature extended-image-insts}}
}

float test_amdgcn_image_sample_lz_2d_f32_f32(float4 v4f32, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_lz_2d_f32_f32(1, f32, f32, tex, vec4i32, 0, 101, 121); //GFX94-error{{'test_amdgcn_image_sample_lz_2d_f32_f32' needs target feature extended-image-insts}}
}

float test_amdgcn_image_sample_l_2d_f32_f32(float4 v4f32, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_l_2d_f32_f32(1, f32, f32, f32, tex, vec4i32, 0, 101, 121); //GFX94-error{{'test_amdgcn_image_sample_l_2d_f32_f32' needs target feature extended-image-insts}}
}

float test_amdgcn_image_sample_d_2d_f32_f32(float4 v4f32, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_d_2d_f32_f32(1, f32, f32, f32, f32, f32, f32, tex, vec4i32, 0, 101, 121); //GFX94-error{{'test_amdgcn_image_sample_d_2d_f32_f32' needs target feature extended-image-insts}}
}

float test_amdgcn_image_sample_lz_2darray_f32_f32(float4 v4f32, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_lz_2darray_f32_f32(1, f32, f32, f32, tex, vec4i32, 0, 101, 121); //GFX94-error{{'test_amdgcn_image_sample_lz_2darray_f32_f32' needs target feature extended-image-insts}}
}

float test_amdgcn_image_sample_l_2darray_f32_f32(float4 v4f32, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_l_2darray_f32_f32(1, f32, f32, f32, f32, tex, vec4i32, 0, 101, 121); //GFX94-error{{'test_amdgcn_image_sample_l_2darray_f32_f32' needs target feature extended-image-insts}}
}

float test_amdgcn_image_sample_d_2darray_f32_f32(float4 v4f32, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_d_2darray_f32_f32(1, f32, f32, f32, f32, f32, f32, f32, tex, vec4i32, 0, 101, 121); //GFX94-error{{'test_amdgcn_image_sample_d_2darray_f32_f32' needs target feature extended-image-insts}}
}
