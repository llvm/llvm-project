// RUN: %clang_cc1 -triple amdgcn-- -target-cpu gfx1100 -S -verify=expected -o - %s
// REQUIRES: amdgpu-registered-target

typedef float float4 __attribute__((ext_vector_type(4)));
typedef half half4 __attribute__((ext_vector_type(4)));

void test_builtin_image_store_2d(float f32, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_store_2d_f32_i32(f32, i32, i32, i32, tex, 106, 103); //expected-error{{argument to '__builtin_amdgcn_image_store_2d_f32_i32' must be a constant integer}}
}
void test_builtin_image_store_2d_1(float4 v4f32, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_store_2d_v4f32_i32(v4f32, 100, i32, i32, tex, i32, 110); //expected-error{{argument to '__builtin_amdgcn_image_store_2d_v4f32_i32' must be a constant integer}}
}
void test_builtin_image_store_2d_2(half4 v4f16, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_store_2d_v4f16_i32(v4f16, 100, i32, i32, tex, 120, i32); //expected-error{{argument to '__builtin_amdgcn_image_store_2d_v4f16_i32' must be a constant integer}}
}

void test_builtin_image_store_2darray(float f32, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_store_2darray_f32_i32(f32, 100, i32, i32, i32, tex, i32, 110); //expected-error{{argument to '__builtin_amdgcn_image_store_2darray_f32_i32' must be a constant integer}}
}
void test_builtin_image_store_2darray_1(float4 v4f32, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_store_2darray_v4f32_i32(v4f32, 100, i32, i32, i32, tex, i32, 110); //expected-error{{argument to '__builtin_amdgcn_image_store_2darray_v4f32_i32' must be a constant integer}}
}
void test_builtin_image_store_2darray_2(half4 v4f16, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_store_2darray_v4f16_i32(v4f16, 100, i32, i32, i32, tex, 120, i32); //expected-error{{argument to '__builtin_amdgcn_image_store_2darray_v4f16_i32' must be a constant integer}}
}

void test_builtin_image_store_1d_1(float4 v4f32, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_store_1d_v4f32_i32(v4f32, i32, i32, tex, 120, i32); //expected-error{{argument to '__builtin_amdgcn_image_store_1d_v4f32_i32' must be a constant integer}}
}
void test_builtin_image_store_1d_2(half4 v4f16, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_store_1d_v4f16_i32(v4f16, 100, i32, tex, 120, i32); //expected-error{{argument to '__builtin_amdgcn_image_store_1d_v4f16_i32' must be a constant integer}}
}

void test_builtin_image_store_1darray_1(float4 v4f32, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_store_1darray_v4f32_i32(v4f32, 100, i32, i32, tex, i32, 110); //expected-error{{argument to '__builtin_amdgcn_image_store_1darray_v4f32_i32' must be a constant integer}}
}
void test_builtin_image_store_1darray_2(half4 v4f16, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_store_1darray_v4f16_i32(v4f16, 100, i32, i32, tex, i32, 110); //expected-error{{argument to '__builtin_amdgcn_image_store_1darray_v4f16_i32' must be a constant integer}}
}

void test_builtin_image_store_3d_1(float4 v4f32, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_store_3d_v4f32_i32(v4f32, 100, i32, i32, i32, tex, 120, i32); //expected-error{{argument to '__builtin_amdgcn_image_store_3d_v4f32_i32' must be a constant integer}}
}
void test_builtin_image_store_3d_2(half4 v4f16, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_store_3d_v4f16_i32(v4f16, i32, i32, i32, i32, tex, 120, i32); //expected-error{{argument to '__builtin_amdgcn_image_store_3d_v4f16_i32' must be a constant integer}}
}

void test_builtin_image_store_cube_1(float4 v4f32, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_store_cube_v4f32_i32(v4f32, i32, i32, i32, i32, tex, 120, 110); //expected-error{{argument to '__builtin_amdgcn_image_store_cube_v4f32_i32' must be a constant integer}}
}
void test_builtin_image_store_cube_2(half4 v4f16, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_store_cube_v4f16_i32(v4f16, i32, i32, i32, i32, tex, 120, 110); //expected-error{{argument to '__builtin_amdgcn_image_store_cube_v4f16_i32' must be a constant integer}}
}

void test_builtin_image_store_mip_1d_1(float4 v4f32, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_store_mip_1d_v4f32_i32(v4f32, i32, i32, i32, tex, 120, i32); //expected-error{{argument to '__builtin_amdgcn_image_store_mip_1d_v4f32_i32' must be a constant integer}}
}
void test_builtin_image_store_mip_1d_2(half4 v4f16, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_store_mip_1d_v4f16_i32(v4f16, 100, i32, i32, tex, 120, i32); //expected-error{{argument to '__builtin_amdgcn_image_store_mip_1d_v4f16_i32' must be a constant integer}}
}

void test_builtin_image_store_mip_1darray_1(float4 v4f32, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_store_mip_1darray_v4f32_i32(v4f32, i32, i32, i32, i32, tex, i32, 110); //expected-error{{argument to '__builtin_amdgcn_image_store_mip_1darray_v4f32_i32' must be a constant integer}}
}
void test_builtin_image_store_mip_1darray_2(half4 v4f16, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_store_mip_1darray_v4f16_i32(v4f16, 100, i32, i32, i32, tex, i32, 110); //expected-error{{argument to '__builtin_amdgcn_image_store_mip_1darray_v4f16_i32' must be a constant integer}}
}

void test_builtin_image_store_mip_2d(float f32, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_store_mip_2d_f32_i32(f32, i32, i32, i32, i32, tex, 120, i32); //expected-error{{argument to '__builtin_amdgcn_image_store_mip_2d_f32_i32' must be a constant integer}}
}
void test_builtin_image_store_mip_2d_1(float4 v4f32, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_store_mip_2d_v4f32_i32(v4f32, 100, i32, i32, i32, tex, 120, i32); //expected-error{{argument to '__builtin_amdgcn_image_store_mip_2d_v4f32_i32' must be a constant integer}}
}
void test_builtin_image_store_mip_2d_2(half4 v4f16, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_store_mip_2d_v4f16_i32(v4f16, i32, i32, i32, i32, tex, 120, 110); //expected-error{{argument to '__builtin_amdgcn_image_store_mip_2d_v4f16_i32' must be a constant integer}}
}

void test_builtin_image_store_mip_2darray(float f32, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_store_mip_2darray_f32_i32(f32, i32, i32, i32, i32, i32, tex, 120, 110); //expected-error{{argument to '__builtin_amdgcn_image_store_mip_2darray_f32_i32' must be a constant integer}}
}
void test_builtin_image_store_mip_2darray_1(float4 v4f32, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_store_mip_2darray_v4f32_i32(v4f32, 100, i32, i32, i32, i32, tex, 120, i32); //expected-error{{argument to '__builtin_amdgcn_image_store_mip_2darray_v4f32_i32' must be a constant integer}}
}
void test_builtin_image_store_mip_2darray_2(half4 v4f16, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_store_mip_2darray_v4f16_i32(v4f16, 100, i32, i32, i32, i32, tex, 120, i32); //expected-error{{argument to '__builtin_amdgcn_image_store_mip_2darray_v4f16_i32' must be a constant integer}}
}

void test_builtin_image_store_mip_3d_1(float4 v4f32, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_store_mip_3d_v4f32_i32(v4f32, i32, i32, i32, i32, i32, tex, i32, 110); //expected-error{{argument to '__builtin_amdgcn_image_store_mip_3d_v4f32_i32' must be a constant integer}}
}
void test_builtin_image_store_mip_3d_2(half4 v4f16, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_store_mip_3d_v4f16_i32(v4f16, i32, i32, i32, i32, i32, tex, i32, 110); //expected-error{{argument to '__builtin_amdgcn_image_store_mip_3d_v4f16_i32' must be a constant integer}}
}

void test_builtin_image_store_mip_cube_1(float4 v4f32, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_store_mip_cube_v4f32_i32(v4f32, i32, i32, i32, i32, i32, tex, 120, i32); //expected-error{{argument to '__builtin_amdgcn_image_store_mip_cube_v4f32_i32' must be a constant integer}}
}
void test_builtin_image_store_mip_cube_2(half4 v4f16, int i32, __amdgpu_texture_t tex) {

  return __builtin_amdgcn_image_store_mip_cube_v4f16_i32(v4f16, 100, i32, i32, i32, i32, tex, 120, i32); //expected-error{{argument to '__builtin_amdgcn_image_store_mip_cube_v4f16_i32' must be a constant integer}}
}
