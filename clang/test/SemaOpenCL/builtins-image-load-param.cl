// RUN: %clang_cc1 -triple amdgcn-- -target-cpu gfx1100 -S -verify=expected -o - %s
// RUN: %clang_cc1 -triple amdgcn-- -target-cpu tahiti -S -verify=expected -o - %s
// RUN: %clang_cc1 -triple amdgcn-- -target-cpu gfx942 -S -verify=GFX94 -o - %s
// REQUIRES: amdgpu-registered-target

typedef int int8 __attribute__((ext_vector_type(8)));
typedef int int4 __attribute__((ext_vector_type(4)));
typedef float float4 __attribute__((ext_vector_type(4)));
typedef half half4 __attribute__((ext_vector_type(4)));

float test_builtin_image_load_2d(float f32, int i32, int8 vec8i32) {

  return __builtin_amdgcn_image_load_2d_f32_i32(i32, i32, i32, vec8i32, 106, 103); //expected-error{{argument to '__builtin_amdgcn_image_load_2d_f32_i32' must be a constant integer}}
}
float4 test_builtin_image_load_2d_1(float4 v4f32, int i32, int8 vec8i32) {

  return __builtin_amdgcn_image_load_2d_v4f32_i32(100, i32, i32, vec8i32, i32, 110); //expected-error{{argument to '__builtin_amdgcn_image_load_2d_v4f32_i32' must be a constant integer}}
}
half4 test_builtin_image_load_2d_2(half4 v4f16, int i32, int8 vec8i32) {

  return __builtin_amdgcn_image_load_2d_v4f16_i32(100, i32, i32, vec8i32, 120, i32); //expected-error{{argument to '__builtin_amdgcn_image_load_2d_v4f16_i32' must be a constant integer}}
}


float test_builtin_image_load_2darray(float f32, int i32, int8 vec8i32) {

  return __builtin_amdgcn_image_load_2darray_f32_i32(100, i32, i32, i32, vec8i32, i32, 110); //expected-error{{argument to '__builtin_amdgcn_image_load_2darray_f32_i32' must be a constant integer}}
}
float4 test_builtin_image_load_2darray_1(float4 v4f32, int i32, int8 vec8i32) {

  return __builtin_amdgcn_image_load_2darray_v4f32_i32(100, i32, i32, i32, vec8i32, i32, 110); //expected-error{{argument to '__builtin_amdgcn_image_load_2darray_v4f32_i32' must be a constant integer}}
}
half4 test_builtin_image_load_2darray_2(half4 v4f16, int i32, int8 vec8i32) {

  return __builtin_amdgcn_image_load_2darray_v4f16_i32(100, i32, i32, i32, vec8i32, 120, i32); //expected-error{{argument to '__builtin_amdgcn_image_load_2darray_v4f16_i32' must be a constant integer}}
}

float4 test_builtin_image_load_1d_1(float4 v4f32, int i32, int8 vec8i32) {

  return __builtin_amdgcn_image_load_1d_v4f32_i32(i32, i32, vec8i32, 120, i32); //expected-error{{argument to '__builtin_amdgcn_image_load_1d_v4f32_i32' must be a constant integer}}
}
half4 test_builtin_image_load_1d_2(half4 v4f16, int i32, int8 vec8i32) {

  return __builtin_amdgcn_image_load_1d_v4f16_i32(100, i32, vec8i32, 120, i32); //expected-error{{argument to '__builtin_amdgcn_image_load_1d_v4f16_i32' must be a constant integer}}
}

float4 test_builtin_image_load_1darray_1(float4 v4f32, int i32, int8 vec8i32) {

  return __builtin_amdgcn_image_load_1darray_v4f32_i32(100, i32, i32, vec8i32, i32, 110); //expected-error{{argument to '__builtin_amdgcn_image_load_1darray_v4f32_i32' must be a constant integer}}
}
half4 test_builtin_image_load_1darray_2(half4 v4f16, int i32, int8 vec8i32) {

  return __builtin_amdgcn_image_load_1darray_v4f16_i32(100, i32, i32, vec8i32, i32, 110); //expected-error{{argument to '__builtin_amdgcn_image_load_1darray_v4f16_i32' must be a constant integer}}
}

float4 test_builtin_image_load_3d_1(float4 v4f32, int i32, int8 vec8i32) {

  return __builtin_amdgcn_image_load_3d_v4f32_i32(100, i32, i32, i32, vec8i32, 120, i32); //expected-error{{argument to '__builtin_amdgcn_image_load_3d_v4f32_i32' must be a constant integer}}
}
half4 test_builtin_image_load_3d_2(half4 v4f16, int i32, int8 vec8i32) {

  return __builtin_amdgcn_image_load_3d_v4f16_i32(i32, i32, i32, i32, vec8i32, 120, i32); //expected-error{{argument to '__builtin_amdgcn_image_load_3d_v4f16_i32' must be a constant integer}}
}

float4 test_builtin_image_load_cube_1(float4 v4f32, int i32, int8 vec8i32) {

  return __builtin_amdgcn_image_load_cube_v4f32_i32(i32, i32, i32, i32, vec8i32, 120, 110); //expected-error{{argument to '__builtin_amdgcn_image_load_cube_v4f32_i32' must be a constant integer}}
}
half4 test_builtin_image_load_cube_2(half4 v4f16, int i32, int8 vec8i32) {

  return __builtin_amdgcn_image_load_cube_v4f16_i32(i32, i32, i32, i32, vec8i32, 120, 110); //expected-error{{argument to '__builtin_amdgcn_image_load_cube_v4f16_i32' must be a constant integer}}
}

float4 test_builtin_image_load_mip_1d_1(float4 v4f32, int i32, int8 vec8i32) {

  return __builtin_amdgcn_image_load_mip_1d_v4f32_i32(i32, i32, i32, vec8i32, 120, i32); //expected-error{{argument to '__builtin_amdgcn_image_load_mip_1d_v4f32_i32' must be a constant integer}}
}
half4 test_builtin_image_load_mip_1d_2(half4 v4f16, int i32, int8 vec8i32) {

  return __builtin_amdgcn_image_load_mip_1d_v4f16_i32(100, i32, i32, vec8i32, 120, i32); //expected-error{{argument to '__builtin_amdgcn_image_load_mip_1d_v4f16_i32' must be a constant integer}}
}

float4 test_builtin_image_load_mip_1darray_1(float4 v4f32, int i32, int8 vec8i32) {

  return __builtin_amdgcn_image_load_mip_1darray_v4f32_i32(i32, i32, i32, i32, vec8i32, i32, 110); //expected-error{{argument to '__builtin_amdgcn_image_load_mip_1darray_v4f32_i32' must be a constant integer}}
}
half4 test_builtin_image_load_mip_1darray_2(half4 v4f16, int i32, int8 vec8i32) {

  return __builtin_amdgcn_image_load_mip_1darray_v4f16_i32(100, i32, i32, i32, vec8i32, i32, 110); //expected-error{{argument to '__builtin_amdgcn_image_load_mip_1darray_v4f16_i32' must be a constant integer}}
}

float test_builtin_image_load_mip_2d(float f32, int i32, int8 vec8i32) {

  return __builtin_amdgcn_image_load_mip_2d_f32_i32(i32, i32, i32, i32, vec8i32, 120, i32); //expected-error{{argument to '__builtin_amdgcn_image_load_mip_2d_f32_i32' must be a constant integer}}
}
float4 test_builtin_image_load_mip_2d_1(float4 v4f32, int i32, int8 vec8i32) {

  return __builtin_amdgcn_image_load_mip_2d_v4f32_i32(100, i32, i32, i32, vec8i32, 120, i32); //expected-error{{argument to '__builtin_amdgcn_image_load_mip_2d_v4f32_i32' must be a constant integer}}
}
half4 test_builtin_image_load_mip_2d_2(half4 v4f16, int i32, int8 vec8i32) {

  return __builtin_amdgcn_image_load_mip_2d_v4f16_i32(i32, i32, i32, i32, vec8i32, 120, 110); //expected-error{{argument to '__builtin_amdgcn_image_load_mip_2d_v4f16_i32' must be a constant integer}}
}

float test_builtin_image_load_mip_2darray(float f32, int i32, int8 vec8i32) {

  return __builtin_amdgcn_image_load_mip_2darray_f32_i32(i32, i32, i32, i32, i32, vec8i32, 120, 110); //expected-error{{argument to '__builtin_amdgcn_image_load_mip_2darray_f32_i32' must be a constant integer}}
}
float4 test_builtin_image_load_mip_2darray_1(float4 v4f32, int i32, int8 vec8i32) {

  return __builtin_amdgcn_image_load_mip_2darray_v4f32_i32(100, i32, i32, i32, i32, vec8i32, 120, i32); //expected-error{{argument to '__builtin_amdgcn_image_load_mip_2darray_v4f32_i32' must be a constant integer}}
}
half4 test_builtin_image_load_mip_2darray_2(half4 v4f16, int i32, int8 vec8i32) {

  return __builtin_amdgcn_image_load_mip_2darray_v4f16_i32(100, i32, i32, i32, i32, vec8i32, 120, i32); //expected-error{{argument to '__builtin_amdgcn_image_load_mip_2darray_v4f16_i32' must be a constant integer}}
}

float4 test_builtin_image_load_mip_3d_1(float4 v4f32, int i32, int8 vec8i32) {

  return __builtin_amdgcn_image_load_mip_3d_v4f32_i32(i32, i32, i32, i32, i32, vec8i32, i32, 110); //expected-error{{argument to '__builtin_amdgcn_image_load_mip_3d_v4f32_i32' must be a constant integer}}
}
half4 test_builtin_image_load_mip_3d_2(half4 v4f16, int i32, int8 vec8i32) {

  return __builtin_amdgcn_image_load_mip_3d_v4f16_i32(i32, i32, i32, i32, i32, vec8i32, i32, 110); //expected-error{{argument to '__builtin_amdgcn_image_load_mip_3d_v4f16_i32' must be a constant integer}}
}

float4 test_builtin_image_load_mip_cube_1(float4 v4f32, int i32, int8 vec8i32) {

  return __builtin_amdgcn_image_load_mip_cube_v4f32_i32(i32, i32, i32, i32, i32, vec8i32, 120, i32); //expected-error{{argument to '__builtin_amdgcn_image_load_mip_cube_v4f32_i32' must be a constant integer}}
}
half4 test_builtin_image_load_mip_cube_2(half4 v4f16, int i32, int8 vec8i32) {

  return __builtin_amdgcn_image_load_mip_cube_v4f16_i32(100, i32, i32, i32, i32, vec8i32, 120, i32); //expected-error{{argument to '__builtin_amdgcn_image_load_mip_cube_v4f16_i32' must be a constant integer}}
}

float test_builtin_image_load_2d_gfx(float f32, int i32, int8 vec8i32) {

  return __builtin_amdgcn_image_load_2d_f32_i32(12, i32, i32, vec8i32, 106, 103); //GFX94-error{{'__builtin_amdgcn_image_load_2d_f32_i32' needs target feature image-insts}}
}
float4 test_builtin_image_load_2d_gfx_1(float4 v4f32, int i32, int8 vec8i32) {

  return __builtin_amdgcn_image_load_2d_v4f32_i32(100, i32, i32, vec8i32, 120, 110); //GFX94-error{{'__builtin_amdgcn_image_load_2d_v4f32_i32' needs target feature image-insts}}
}
half4 test_builtin_image_load_2d_gfx_2(half4 v4f16, int i32, int8 vec8i32) {

  return __builtin_amdgcn_image_load_2d_v4f16_i32(100, i32, i32, vec8i32, 120, 110); //GFX94-error{{'__builtin_amdgcn_image_load_2d_v4f16_i32' needs target feature image-insts}}
}

float test_builtin_image_sample_2d(float f32, int i32, int8 vec8i32, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_2d_f32_f32(i32, f32, f32, vec8i32, vec4i32, 0, 106, 103); //expected-error{{argument to '__builtin_amdgcn_image_sample_2d_f32_f32' must be a constant integer}}
}
float4 test_builtin_image_sample_2d_1(float4 v4f32, float f32, int i32, int8 vec8i32, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_2d_v4f32_f32(100, f32, f32, vec8i32, vec4i32, 0, i32, 110); //expected-error{{argument to '__builtin_amdgcn_image_sample_2d_v4f32_f32' must be a constant integer}}
}
half4 test_builtin_image_sample_2d_2(half4 v4f16, float f32, int i32, int8 vec8i32, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_2d_v4f16_f32(100, f32, f32, vec8i32, vec4i32, 0, 120, i32); //expected-error{{argument to '__builtin_amdgcn_image_sample_2d_v4f16_f32' must be a constant integer}}
}

float test_builtin_image_sample_2darray(float f32, int i32, int8 vec8i32, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_2darray_f32_f32(100, f32, f32, f32, vec8i32, vec4i32, 0, i32, 110); //expected-error{{argument to '__builtin_amdgcn_image_sample_2darray_f32_f32' must be a constant integer}}
}
float4 test_builtin_image_sample_2darray_1(float4 v4f32, float f32, int i32, int8 vec8i32, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_2darray_v4f32_f32(100, f32, f32, f32, vec8i32, vec4i32, 0, i32, 110); //expected-error{{argument to '__builtin_amdgcn_image_sample_2darray_v4f32_f32' must be a constant integer}}
}
half4 test_builtin_image_sample_2darray_2(half4 v4f16, float f32, int i32, int8 vec8i32, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_2darray_v4f16_f32(100, f32, f32, f32, vec8i32, vec4i32, 0, 120, i32); //expected-error{{argument to '__builtin_amdgcn_image_sample_2darray_v4f16_f32' must be a constant integer}}
}

float4 test_builtin_image_sample_1d_1(float4 v4f32, float f32, int i32, int8 vec8i32, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_1d_v4f32_f32(i32, f32, vec8i32, vec4i32, 0, 120, i32); //expected-error{{argument to '__builtin_amdgcn_image_sample_1d_v4f32_f32' must be a constant integer}}
}
half4 test_builtin_image_sample_1d_2(half4 v4f16, float f32, int i32, int8 vec8i32, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_1d_v4f16_f32(100, f32, vec8i32, vec4i32, 0, 120, i32); //expected-error{{argument to '__builtin_amdgcn_image_sample_1d_v4f16_f32' must be a constant integer}}
}

float4 test_builtin_image_sample_1darray_1(float4 v4f32, float f32, int i32, int8 vec8i32, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_1darray_v4f32_f32(100, f32, f32, vec8i32, vec4i32, 0, i32, 110); //expected-error{{argument to '__builtin_amdgcn_image_sample_1darray_v4f32_f32' must be a constant integer}}
}
half4 test_builtin_image_sample_1darray_2(half4 v4f16, float f32, int i32, int8 vec8i32, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_1darray_v4f16_f32(100, f32, f32, vec8i32, vec4i32, 0, i32, 110); //expected-error{{argument to '__builtin_amdgcn_image_sample_1darray_v4f16_f32' must be a constant integer}}
}

float4 test_builtin_image_sample_3d_1(float4 v4f32, float f32, int i32, int8 vec8i32, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_3d_v4f32_f32(100, f32, f32, f32, vec8i32, vec4i32, 0, 120, i32); //expected-error{{argument to '__builtin_amdgcn_image_sample_3d_v4f32_f32' must be a constant integer}}
}
half4 test_builtin_image_sample_3d_2(half4 v4f16, float f32, int i32, int8 vec8i32, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_3d_v4f16_f32(i32, f32, f32, f32, vec8i32, vec4i32, 0, 120, i32); //expected-error{{argument to '__builtin_amdgcn_image_sample_3d_v4f16_f32' must be a constant integer}}
}

float4 test_builtin_image_sample_cube_1(float4 v4f32, float f32, int i32, int8 vec8i32, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_cube_v4f32_f32(i32, f32, f32, f32, vec8i32, vec4i32, 0, 120, 110); //expected-error{{argument to '__builtin_amdgcn_image_sample_cube_v4f32_f32' must be a constant integer}}
}
half4 test_builtin_image_sample_cube_2(half4 v4f16, float f32, int i32, int8 vec8i32, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_cube_v4f16_f32(i32, f32, f32, f32, vec8i32, vec4i32, 0, 120, 110); //expected-error{{argument to '__builtin_amdgcn_image_sample_cube_v4f16_f32' must be a constant integer}}
}

float test_builtin_image_sample_2d_gfx(float f32, int i32, int8 vec8i32, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_2d_f32_f32(100, f32, f32, vec8i32, vec4i32, 0, 120, 110); //GFX94-error{{'__builtin_amdgcn_image_sample_2d_f32_f32' needs target feature image-insts}}
}
float4 test_builtin_image_sample_2d_gfx_1(float4 v4f32, float f32, int i32, int8 vec8i32, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_2d_v4f32_f32(100, f32, f32, vec8i32, vec4i32, 0, 120, 110); //GFX94-error{{'__builtin_amdgcn_image_sample_2d_v4f32_f32' needs target feature image-insts}}
}
half4 test_builtin_image_sample_2d_gfx_2(half4 v4f16, float f32, int i32, int8 vec8i32, int4 vec4i32) {

  return __builtin_amdgcn_image_sample_2d_v4f16_f32(100, f32, f32, vec8i32, vec4i32, 0, 120, 110); //GFX94-error{{'__builtin_amdgcn_image_sample_2d_v4f16_f32' needs target feature image-insts}}
}
