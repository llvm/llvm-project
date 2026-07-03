// RUN: %clang_cc1 -triple amdgcn-- -target-cpu gfx1100 -S -verify=expected -o - %s
// REQUIRES: amdgpu-registered-target

typedef int int4 __attribute__((ext_vector_type(4)));
typedef float float4 __attribute__((ext_vector_type(4)));
typedef half half4 __attribute__((ext_vector_type(4)));

float test_builtin_image_load_2d(float f32, int i32, __amdgpu_texture_t tex) {

  float a = __builtin_amdgcn_image_load_2d_f32_i32(i32, i32, i32, tex, 106, 103); //expected-error{{argument to '__builtin_amdgcn_image_load_2d_f32_i32' must be a constant integer}}
  return a + __builtin_amdgcn_image_load_2d_f32_i32(5, i32, i32, tex, 106, 103); //expected-error{{dmask argument cannot have more bits set than there are elements in return type}}
}
float4 test_builtin_image_load_2d_1(float4 v4f32, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_load_2d_v4f32_i32(15, i32, i32, tex, i32, 110); //expected-error{{argument to '__builtin_amdgcn_image_load_2d_v4f32_i32' must be a constant integer}}
}
half4 test_builtin_image_load_2d_2(half4 v4f16, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_load_2d_v4f16_i32(15, i32, i32, tex, 120, i32); //expected-error{{argument to '__builtin_amdgcn_image_load_2d_v4f16_i32' must be a constant integer}}
}
half4 test_builtin_image_load_2d_dmask_range(half4 v4f16, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_load_2d_v4f16_i32(100, i32, i32, tex, 120, 110); //expected-error{{argument value 100 is outside the valid range [0, 15]}}
}


float test_builtin_image_load_2darray(float f32, int i32, __amdgpu_texture_t tex) {

  float a =  __builtin_amdgcn_image_load_2darray_f32_i32(1, i32, i32, i32, tex, i32, 110); //expected-error{{argument to '__builtin_amdgcn_image_load_2darray_f32_i32' must be a constant integer}}
  return a + __builtin_amdgcn_image_load_2darray_f32_i32(3, i32, i32, i32, tex, i32, 110); //expected-error{{dmask argument cannot have more bits set than there are elements in return type}}
}
float4 test_builtin_image_load_2darray_1(float4 v4f32, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_load_2darray_v4f32_i32(15, i32, i32, i32, tex, i32, 110); //expected-error{{argument to '__builtin_amdgcn_image_load_2darray_v4f32_i32' must be a constant integer}}
}
half4 test_builtin_image_load_2darray_2(half4 v4f16, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_load_2darray_v4f16_i32(15, i32, i32, i32, tex, 120, i32); //expected-error{{argument to '__builtin_amdgcn_image_load_2darray_v4f16_i32' must be a constant integer}}
}
half4 test_builtin_image_load_2darray_dmask_range(half4 v4f16, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_load_2darray_v4f16_i32(100, i32, i32, i32, tex, 120, 110); //expected-error{{argument value 100 is outside the valid range [0, 15]}}
}

float4 test_builtin_image_load_1d_1(float4 v4f32, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_load_1d_v4f32_i32(i32, i32, tex, 120, i32); //expected-error{{argument to '__builtin_amdgcn_image_load_1d_v4f32_i32' must be a constant integer}}
}
half4 test_builtin_image_load_1d_2(half4 v4f16, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_load_1d_v4f16_i32(15, i32, tex, 120, i32); //expected-error{{argument to '__builtin_amdgcn_image_load_1d_v4f16_i32' must be a constant integer}}
}
half4 test_builtin_image_load_1d_dmask_range(half4 v4f16, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_load_1d_v4f16_i32(100, i32, tex, 120, 110); //expected-error{{argument value 100 is outside the valid range [0, 15]}}
}

float4 test_builtin_image_load_1darray_1(float4 v4f32, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_load_1darray_v4f32_i32(15, i32, i32, tex, i32, 110); //expected-error{{argument to '__builtin_amdgcn_image_load_1darray_v4f32_i32' must be a constant integer}}
}
half4 test_builtin_image_load_1darray_2(half4 v4f16, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_load_1darray_v4f16_i32(15, i32, i32, tex, i32, 110); //expected-error{{argument to '__builtin_amdgcn_image_load_1darray_v4f16_i32' must be a constant integer}}
}
half4 test_builtin_image_load_1darray_dmask_range(half4 v4f16, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_load_1darray_v4f16_i32(100, i32, i32, tex, 120, 110); //expected-error{{argument value 100 is outside the valid range [0, 15]}}
}

float4 test_builtin_image_load_3d_1(float4 v4f32, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_load_3d_v4f32_i32(15, i32, i32, i32, tex, 120, i32); //expected-error{{argument to '__builtin_amdgcn_image_load_3d_v4f32_i32' must be a constant integer}}
}
half4 test_builtin_image_load_3d_2(half4 v4f16, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_load_3d_v4f16_i32(i32, i32, i32, i32, tex, 120, i32); //expected-error{{argument to '__builtin_amdgcn_image_load_3d_v4f16_i32' must be a constant integer}}
}
half4 test_builtin_image_load_3d_dmask_range(half4 v4f16, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_load_3d_v4f16_i32(100, i32, i32, i32, tex, 120, 110); //expected-error{{argument value 100 is outside the valid range [0, 15]}}
}

float4 test_builtin_image_load_cube_1(float4 v4f32, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_load_cube_v4f32_i32(i32, i32, i32, i32, tex, 120, 110); //expected-error{{argument to '__builtin_amdgcn_image_load_cube_v4f32_i32' must be a constant integer}}
}
half4 test_builtin_image_load_cube_2(half4 v4f16, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_load_cube_v4f16_i32(i32, i32, i32, i32, tex, 120, 110); //expected-error{{argument to '__builtin_amdgcn_image_load_cube_v4f16_i32' must be a constant integer}}
}
half4 test_builtin_image_load_cube_dmask_range(half4 v4f16, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_load_cube_v4f16_i32(100, i32, i32, i32, tex, 120, 110); //expected-error{{argument value 100 is outside the valid range [0, 15]}}
}

float4 test_builtin_image_load_mip_1d_1(float4 v4f32, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_load_mip_1d_v4f32_i32(i32, i32, i32, tex, 120, i32); //expected-error{{argument to '__builtin_amdgcn_image_load_mip_1d_v4f32_i32' must be a constant integer}}
}
half4 test_builtin_image_load_mip_1d_2(half4 v4f16, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_load_mip_1d_v4f16_i32(15, i32, i32, tex, 120, i32); //expected-error{{argument to '__builtin_amdgcn_image_load_mip_1d_v4f16_i32' must be a constant integer}}
}
half4 test_builtin_image_load_mip_1d_dmask_range(half4 v4f16, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_load_mip_1d_v4f16_i32(100, i32, i32, tex, 120, 110); //expected-error{{argument value 100 is outside the valid range [0, 15]}}
}

float4 test_builtin_image_load_mip_1darray_1(float4 v4f32, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_load_mip_1darray_v4f32_i32(i32, i32, i32, i32, tex, i32, 110); //expected-error{{argument to '__builtin_amdgcn_image_load_mip_1darray_v4f32_i32' must be a constant integer}}
}
half4 test_builtin_image_load_mip_1darray_2(half4 v4f16, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_load_mip_1darray_v4f16_i32(15, i32, i32, i32, tex, i32, 110); //expected-error{{argument to '__builtin_amdgcn_image_load_mip_1darray_v4f16_i32' must be a constant integer}}
}
half4 test_builtin_image_load_mip_1darray_dmask_range(half4 v4f16, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_load_mip_1darray_v4f16_i32(100, i32, i32, i32, tex, 120, 110); //expected-error{{argument value 100 is outside the valid range [0, 15]}}
}

float test_builtin_image_load_mip_2d(float f32, int i32, __amdgpu_texture_t tex) {

  float a = __builtin_amdgcn_image_load_mip_2d_f32_i32(i32, i32, i32, i32, tex, 120, i32); //expected-error{{argument to '__builtin_amdgcn_image_load_mip_2d_f32_i32' must be a constant integer}}
  return a + __builtin_amdgcn_image_load_mip_2d_f32_i32(7, i32, i32, i32, tex, 120, 110); //expected-error{{dmask argument cannot have more bits set than there are elements in return type}}
}
float4 test_builtin_image_load_mip_2d_1(float4 v4f32, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_load_mip_2d_v4f32_i32(15, i32, i32, i32, tex, 120, i32); //expected-error{{argument to '__builtin_amdgcn_image_load_mip_2d_v4f32_i32' must be a constant integer}}
}
half4 test_builtin_image_load_mip_2d_2(half4 v4f16, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_load_mip_2d_v4f16_i32(i32, i32, i32, i32, tex, 120, 110); //expected-error{{argument to '__builtin_amdgcn_image_load_mip_2d_v4f16_i32' must be a constant integer}}
}
half4 test_builtin_image_load_mip_2d_dmask_range(half4 v4f16, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_load_mip_2d_v4f16_i32(100, i32, i32, i32, tex, 120, 110); //expected-error{{argument value 100 is outside the valid range [0, 15]}}
}

float test_builtin_image_load_mip_2darray(float f32, int i32, __amdgpu_texture_t tex) {

  float a = __builtin_amdgcn_image_load_mip_2darray_f32_i32(i32, i32, i32, i32, i32, tex, 120, 110); //expected-error{{argument to '__builtin_amdgcn_image_load_mip_2darray_f32_i32' must be a constant integer}}
  return a + __builtin_amdgcn_image_load_mip_2darray_f32_i32(9, i32, i32, i32, i32, tex, 120, 110); //expected-error{{dmask argument cannot have more bits set than there are elements in return type}}
}
float4 test_builtin_image_load_mip_2darray_1(float4 v4f32, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_load_mip_2darray_v4f32_i32(15, i32, i32, i32, i32, tex, 120, i32); //expected-error{{argument to '__builtin_amdgcn_image_load_mip_2darray_v4f32_i32' must be a constant integer}}
}
half4 test_builtin_image_load_mip_2darray_2(half4 v4f16, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_load_mip_2darray_v4f16_i32(15, i32, i32, i32, i32, tex, 120, i32); //expected-error{{argument to '__builtin_amdgcn_image_load_mip_2darray_v4f16_i32' must be a constant integer}}
}
half4 test_builtin_image_load_mip_2darray_dmask_range(half4 v4f16, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_load_mip_2darray_v4f16_i32(100, i32, i32, i32, i32, tex, 120, 110); //expected-error{{argument value 100 is outside the valid range [0, 15]}}
}

float4 test_builtin_image_load_mip_3d_1(float4 v4f32, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_load_mip_3d_v4f32_i32(i32, i32, i32, i32, i32, tex, i32, 110); //expected-error{{argument to '__builtin_amdgcn_image_load_mip_3d_v4f32_i32' must be a constant integer}}
}
half4 test_builtin_image_load_mip_3d_2(half4 v4f16, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_load_mip_3d_v4f16_i32(i32, i32, i32, i32, i32, tex, i32, 110); //expected-error{{argument to '__builtin_amdgcn_image_load_mip_3d_v4f16_i32' must be a constant integer}}
}
half4 test_builtin_image_load_mip_3d_dmask_range(half4 v4f16, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_load_mip_3d_v4f16_i32(100, i32, i32, i32, i32, tex, 120, 110); //expected-error{{argument value 100 is outside the valid range [0, 15]}}
}

float4 test_builtin_image_load_mip_cube_1(float4 v4f32, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_load_mip_cube_v4f32_i32(i32, i32, i32, i32, i32, tex, 120, i32); //expected-error{{argument to '__builtin_amdgcn_image_load_mip_cube_v4f32_i32' must be a constant integer}}
}
half4 test_builtin_image_load_mip_cube_2(half4 v4f16, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_load_mip_cube_v4f16_i32(15, i32, i32, i32, i32, tex, 120, i32); //expected-error{{argument to '__builtin_amdgcn_image_load_mip_cube_v4f16_i32' must be a constant integer}}
}
half4 test_builtin_image_load_mip_cube_dmask_range(half4 v4f16, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_load_mip_cube_v4f16_i32(100, i32, i32, i32, i32, tex, 120, 110); //expected-error{{argument value 100 is outside the valid range [0, 15]}}
}

float test_builtin_image_sample_2d(float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  float a = __builtin_amdgcn_image_sample_2d_f32_f32(i32, f32, f32, tex, vec4i32, 0, 106, 103); //expected-error{{argument to '__builtin_amdgcn_image_sample_2d_f32_f32' must be a constant integer}}
  return a + __builtin_amdgcn_image_sample_2d_f32_f32(15, f32, f32, tex, vec4i32, 0, 106, 103); //expected-error{{dmask argument cannot have more bits set than there are elements in return type}}
}
float4 test_builtin_image_sample_2d_1(float4 v4f32, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_2d_v4f32_f32(15, f32, f32, tex, vec4i32, 0, i32, 110); //expected-error{{argument to '__builtin_amdgcn_image_sample_2d_v4f32_f32' must be a constant integer}}
}
half4 test_builtin_image_sample_2d_2(half4 v4f16, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_2d_v4f16_f32(15, f32, f32, tex, vec4i32, 0, 120, i32); //expected-error{{argument to '__builtin_amdgcn_image_sample_2d_v4f16_f32' must be a constant integer}}
}
half4 test_builtin_image_sample_2d_dmask_range(half4 v4f16, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_2d_v4f16_f32(100, f32, f32, tex, vec4i32, 0, 120, 110); //expected-error{{argument value 100 is outside the valid range [0, 15]}}
}

float test_builtin_image_sample_2darray(float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  float a = __builtin_amdgcn_image_sample_2darray_f32_f32(1, f32, f32, f32, tex, vec4i32, 0, i32, 110); //expected-error{{argument to '__builtin_amdgcn_image_sample_2darray_f32_f32' must be a constant integer}}
  return a + __builtin_amdgcn_image_sample_2darray_f32_f32(11, f32, f32, f32, tex, vec4i32, 0, 120, 110); //expected-error{{dmask argument cannot have more bits set than there are elements in return type}}
}
float4 test_builtin_image_sample_2darray_1(float4 v4f32, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_2darray_v4f32_f32(15, f32, f32, f32, tex, vec4i32, 0, i32, 110); //expected-error{{argument to '__builtin_amdgcn_image_sample_2darray_v4f32_f32' must be a constant integer}}
}
half4 test_builtin_image_sample_2darray_2(half4 v4f16, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_2darray_v4f16_f32(15, f32, f32, f32, tex, vec4i32, 0, 120, i32); //expected-error{{argument to '__builtin_amdgcn_image_sample_2darray_v4f16_f32' must be a constant integer}}
}
half4 test_builtin_image_sample_2darray_dmask_range(half4 v4f16, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_2darray_v4f16_f32(100, f32, f32, f32, tex, vec4i32, 0, 120, 110); //expected-error{{argument value 100 is outside the valid range [0, 15]}}
}

float4 test_builtin_image_sample_1d_1(float4 v4f32, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_1d_v4f32_f32(i32, f32, tex, vec4i32, 0, 120, i32); //expected-error{{argument to '__builtin_amdgcn_image_sample_1d_v4f32_f32' must be a constant integer}}
}
half4 test_builtin_image_sample_1d_2(half4 v4f16, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_1d_v4f16_f32(15, f32, tex, vec4i32, 0, 120, i32); //expected-error{{argument to '__builtin_amdgcn_image_sample_1d_v4f16_f32' must be a constant integer}}
}
half4 test_builtin_image_sample_1d_dmask_range(half4 v4f16, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_1d_v4f16_f32(100, f32, tex, vec4i32, 0, 120, 110); //expected-error{{argument value 100 is outside the valid range [0, 15]}}
}

float4 test_builtin_image_sample_1darray_1(float4 v4f32, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_1darray_v4f32_f32(15, f32, f32, tex, vec4i32, 0, i32, 110); //expected-error{{argument to '__builtin_amdgcn_image_sample_1darray_v4f32_f32' must be a constant integer}}
}
half4 test_builtin_image_sample_1darray_2(half4 v4f16, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_1darray_v4f16_f32(15, f32, f32, tex, vec4i32, 0, i32, 110); //expected-error{{argument to '__builtin_amdgcn_image_sample_1darray_v4f16_f32' must be a constant integer}}
}
half4 test_builtin_image_sample_1darray_dmask_range(half4 v4f16, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_1darray_v4f16_f32(100, f32, f32, tex, vec4i32, 0, 120, 110); //expected-error{{argument value 100 is outside the valid range [0, 15]}}
}

float4 test_builtin_image_sample_3d_1(float4 v4f32, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_3d_v4f32_f32(15, f32, f32, f32, tex, vec4i32, 0, 120, i32); //expected-error{{argument to '__builtin_amdgcn_image_sample_3d_v4f32_f32' must be a constant integer}}
}
half4 test_builtin_image_sample_3d_2(half4 v4f16, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_3d_v4f16_f32(i32, f32, f32, f32, tex, vec4i32, 0, 120, i32); //expected-error{{argument to '__builtin_amdgcn_image_sample_3d_v4f16_f32' must be a constant integer}}
}
half4 test_builtin_image_sample_3d_dmask_range(half4 v4f16, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_3d_v4f16_f32(100, f32, f32, f32, tex, vec4i32, 0, 120, 110); //expected-error{{argument value 100 is outside the valid range [0, 15]}}
}

float4 test_builtin_image_sample_cube_1(float4 v4f32, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_cube_v4f32_f32(i32, f32, f32, f32, tex, vec4i32, 0, 120, 110); //expected-error{{argument to '__builtin_amdgcn_image_sample_cube_v4f32_f32' must be a constant integer}}
}
half4 test_builtin_image_sample_cube_2(half4 v4f16, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_cube_v4f16_f32(i32, f32, f32, f32, tex, vec4i32, 0, 120, 110); //expected-error{{argument to '__builtin_amdgcn_image_sample_cube_v4f16_f32' must be a constant integer}}
}
half4 test_builtin_image_sample_cube_dmask_range(half4 v4f16, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_cube_v4f16_f32(100, f32, f32, f32, tex, vec4i32, 0, 120, 110); //expected-error{{argument value 100 is outside the valid range [0, 15]}}
}
