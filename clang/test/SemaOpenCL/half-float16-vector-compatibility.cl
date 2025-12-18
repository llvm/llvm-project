// RUN: %clang_cc1 -triple amdgcn-- -target-cpu gfx1100 -verify -S -o - %s
// REQUIRES: amdgpu-registered-target

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

typedef int int4 __attribute__((ext_vector_type(4)));
typedef float float4 __attribute__((ext_vector_type(4)));

typedef half half2 __attribute__((ext_vector_type(2)));
typedef half half3 __attribute__((ext_vector_type(3)));
typedef half half4 __attribute__((ext_vector_type(4)));
typedef half half8 __attribute__((ext_vector_type(8)));
typedef half half16 __attribute__((ext_vector_type(16)));

typedef _Float16 float16_2 __attribute__((ext_vector_type(2)));
typedef _Float16 float16_3 __attribute__((ext_vector_type(3)));
typedef _Float16 float16_4 __attribute__((ext_vector_type(4)));
typedef _Float16 float16_8 __attribute__((ext_vector_type(8)));
typedef _Float16 float16_16 __attribute__((ext_vector_type(16)));

void test_half_vector_to_float16(float16_2 f16_2, float16_3 f16_3, float16_4 f16_4, float16_8 f16_8, float16_16 f16_16) {
  half2 h2 = f16_2; // expected-no-error
  half3 h3 = f16_3; // expected-no-error
  half4 h4 = f16_4; // expected-no-error
  half8 h8 = f16_8; // expected-no-error
  half16 h16 = f16_16; // expected-no-error
}

void test_float16_vector_to_half(half2 h2, half3 h3, half4 h4, half8 h8, half16 h16) {
  float16_2 f16_2 = h2; // expected-no-error
  float16_3 f16_3 = h3; // expected-no-error
  float16_4 f16_4 = h4; // expected-no-error
  float16_8 f16_8 = h8; // expected-no-error
  float16_16 f16_16 = h16; // expected-no-error
}

half4 test_return_half4_from_float16_vector(float16_4 f16_4) {
  return f16_4; // expected-no-error
}

float16_4 test_return_float16_4_from_half4(half4 h4) {
  return h4; // expected-no-error
}

half4 test_explicit_cast_half4_to_float16_vector(half4 h4) {
  return (float16_4)h4; // expected-no-error
}

float16_4 test_explicit_cast_float16_4_to_half4(float16_4 f16_4) {
  return (half4)f16_4; // expected-no-error
}

half4 test_builtin_image_load_2d_2(half4 v4f16, int i32, __amdgpu_texture_t tex) {
  return __builtin_amdgcn_image_load_2d_v4f16_i32(100, i32, i32, tex, 120, 110); // expected-no-error
}

half4 test_builtin_amdgcn_image_sample_2d_v4f16_f32(half4 v4f16, int i32, float f32, __amdgpu_texture_t tex, int4 vec4i32) {
  return __builtin_amdgcn_image_sample_2d_v4f16_f32(100, f32, f32, tex, vec4i32, 0, 120, 110); // expected-no-error
}

void test_half_mismatch_vector_size_error(float16_2 f16_2, float16_3 f16_3, float16_4 f16_4, float16_8 f16_8, float16_16 f16_16) {
  half2 h2 = f16_3  ; // expected-error{{initializing '__private half2' (vector of 2 'half' values) with an expression of incompatible type '__private float16_3' (vector of 3 '_Float16' values)}}
  half3 h3 = f16_2; // expected-error{{initializing '__private half3' (vector of 3 'half' values) with an expression of incompatible type '__private float16_2' (vector of 2 '_Float16' values)}}
  half4 h4 = f16_8; // expected-error{{initializing '__private half4' (vector of 4 'half' values) with an expression of incompatible type '__private float16_8' (vector of 8 '_Float16' values)}}
  half8 h8 = f16_4; // expected-error{{initializing '__private half8' (vector of 8 'half' values) with an expression of incompatible type '__private float16_4' (vector of 4 '_Float16' values)}}
  half16 h16 = f16_4; // expected-error{{initializing '__private half16' (vector of 16 'half' values) with an expression of incompatible type '__private float16_4' (vector of 4 '_Float16' values)}}
}

void test_float16_mismatch_vector_size_error(half2 h2, half3 h3, half4 h4, half8 h8, half16 h16) {
  float16_2 f16_2 = h3; // expected-error{{initializing '__private float16_2' (vector of 2 '_Float16' values) with an expression of incompatible type '__private half3' (vector of 3 'half' values)}}
  float16_3 f16_3 = h2; // expected-error{{initializing '__private float16_3' (vector of 3 '_Float16' values) with an expression of incompatible type '__private half2' (vector of 2 'half' values)}}
  float16_4 f16_4 = h8; // expected-error{{initializing '__private float16_4' (vector of 4 '_Float16' values) with an expression of incompatible type '__private half8' (vector of 8 'half' values)}}
  float16_8 f16_8 = h4; // expected-error{{initializing '__private float16_8' (vector of 8 '_Float16' values) with an expression of incompatible type '__private half4' (vector of 4 'half' values)}}
  float16_16 f16_16 = h4; // expected-error{{initializing '__private float16_16' (vector of 16 '_Float16' values) with an expression of incompatible type '__private half4' (vector of 4 'half' values)}}
}
