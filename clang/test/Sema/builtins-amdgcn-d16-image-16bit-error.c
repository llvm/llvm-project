// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgpu7.00-unknown-unknown -verify -fsyntax-only %s

// Verify that half typed image intrinsics require 16-bit-insts as well as
// image-insts.

typedef _Float16 half;
typedef half half4 __attribute__((ext_vector_type(4)));
typedef int int4 __attribute__((ext_vector_type(4)));

void test(half4 v, __amdgpu_texture_t tex) {
  v = __builtin_amdgcn_image_load_1d_v4f16_i32( // expected-error {{needs target feature image-insts,16-bit-insts}}
      0, 0, tex, 0, 0);
  v = __builtin_amdgcn_image_load_1darray_v4f16_i32( // expected-error {{needs target feature image-insts,16-bit-insts}}
      0, 0, 0, tex, 0, 0);
  v = __builtin_amdgcn_image_load_2d_v4f16_i32( // expected-error {{needs target feature image-insts,16-bit-insts}}
      0, 0, 0, tex, 0, 0);
  v = __builtin_amdgcn_image_load_2darray_v4f16_i32( // expected-error {{needs target feature image-insts,16-bit-insts}}
      0, 0, 0, 0, tex, 0, 0);
  v = __builtin_amdgcn_image_load_3d_v4f16_i32( // expected-error {{needs target feature image-insts,16-bit-insts}}
      0, 0, 0, 0, tex, 0, 0);
  v = __builtin_amdgcn_image_load_cube_v4f16_i32( // expected-error {{needs target feature image-insts,16-bit-insts}}
      0, 0, 0, 0, tex, 0, 0);
  v = __builtin_amdgcn_image_load_mip_1d_v4f16_i32( // expected-error {{needs target feature image-insts,16-bit-insts}}
      0, 0, 0, tex, 0, 0);
  v = __builtin_amdgcn_image_load_mip_1darray_v4f16_i32( // expected-error {{needs target feature image-insts,16-bit-insts}}
      0, 0, 0, 0, tex, 0, 0);
  v = __builtin_amdgcn_image_load_mip_2d_v4f16_i32( // expected-error {{needs target feature image-insts,16-bit-insts}}
      0, 0, 0, 0, tex, 0, 0);
  v = __builtin_amdgcn_image_load_mip_2darray_v4f16_i32( // expected-error {{needs target feature image-insts,16-bit-insts}}
      0, 0, 0, 0, 0, tex, 0, 0);
  v = __builtin_amdgcn_image_load_mip_3d_v4f16_i32( // expected-error {{needs target feature image-insts,16-bit-insts}}
      0, 0, 0, 0, 0, tex, 0, 0);
  v = __builtin_amdgcn_image_load_mip_cube_v4f16_i32( // expected-error {{needs target feature image-insts,16-bit-insts}}
      0, 0, 0, 0, 0, tex, 0, 0);
  __builtin_amdgcn_image_store_1d_v4f16_i32( // expected-error {{needs target feature image-insts,16-bit-insts}}
      v, 0, 0, tex, 0, 0);
  __builtin_amdgcn_image_store_1darray_v4f16_i32( // expected-error {{needs target feature image-insts,16-bit-insts}}
      v, 0, 0, 0, tex, 0, 0);
  __builtin_amdgcn_image_store_2d_v4f16_i32( // expected-error {{needs target feature image-insts,16-bit-insts}}
      v, 0, 0, 0, tex, 0, 0);
  __builtin_amdgcn_image_store_2darray_v4f16_i32( // expected-error {{needs target feature image-insts,16-bit-insts}}
      v, 0, 0, 0, 0, tex, 0, 0);
  __builtin_amdgcn_image_store_3d_v4f16_i32( // expected-error {{needs target feature image-insts,16-bit-insts}}
      v, 0, 0, 0, 0, tex, 0, 0);
  __builtin_amdgcn_image_store_cube_v4f16_i32( // expected-error {{needs target feature image-insts,16-bit-insts}}
      v, 0, 0, 0, 0, tex, 0, 0);
  __builtin_amdgcn_image_store_mip_1d_v4f16_i32( // expected-error {{needs target feature image-insts,16-bit-insts}}
      v, 0, 0, 0, tex, 0, 0);
  __builtin_amdgcn_image_store_mip_1darray_v4f16_i32( // expected-error {{needs target feature image-insts,16-bit-insts}}
      v, 0, 0, 0, 0, tex, 0, 0);
  __builtin_amdgcn_image_store_mip_2d_v4f16_i32( // expected-error {{needs target feature image-insts,16-bit-insts}}
      v, 0, 0, 0, 0, tex, 0, 0);
  __builtin_amdgcn_image_store_mip_2darray_v4f16_i32( // expected-error {{needs target feature image-insts,16-bit-insts}}
      v, 0, 0, 0, 0, 0, tex, 0, 0);
  __builtin_amdgcn_image_store_mip_3d_v4f16_i32( // expected-error {{needs target feature image-insts,16-bit-insts}}
      v, 0, 0, 0, 0, 0, tex, 0, 0);
  __builtin_amdgcn_image_store_mip_cube_v4f16_i32( // expected-error {{needs target feature image-insts,16-bit-insts}}
      v, 0, 0, 0, 0, 0, tex, 0, 0);
}

void test_sample(half4 v, float f32, __amdgpu_texture_t tex, int4 vec4i32) {
  v = __builtin_amdgcn_image_sample_1d_v4f16_f32( // expected-error {{needs target feature image-insts,16-bit-insts}}
      15, f32, tex, vec4i32, 0, 0, 0);
  v = __builtin_amdgcn_image_sample_1darray_v4f16_f32( // expected-error {{needs target feature image-insts,16-bit-insts}}
      15, f32, f32, tex, vec4i32, 0, 0, 0);
  v = __builtin_amdgcn_image_sample_2d_v4f16_f32( // expected-error {{needs target feature image-insts,16-bit-insts}}
      15, f32, f32, tex, vec4i32, 0, 0, 0);
  v = __builtin_amdgcn_image_sample_2darray_v4f16_f32( // expected-error {{needs target feature image-insts,16-bit-insts}}
      15, f32, f32, f32, tex, vec4i32, 0, 0, 0);
  v = __builtin_amdgcn_image_sample_3d_v4f16_f32( // expected-error {{needs target feature image-insts,16-bit-insts}}
      15, f32, f32, f32, tex, vec4i32, 0, 0, 0);
  v = __builtin_amdgcn_image_sample_cube_v4f16_f32( // expected-error {{needs target feature image-insts,16-bit-insts}}
      15, f32, f32, f32, tex, vec4i32, 0, 0, 0);
  v = __builtin_amdgcn_image_sample_lz_1d_v4f16_f32( // expected-error {{needs target feature extended-image-insts,16-bit-insts}}
      15, f32, tex, vec4i32, 0, 0, 0);
  v = __builtin_amdgcn_image_sample_lz_1darray_v4f16_f32( // expected-error {{needs target feature extended-image-insts,16-bit-insts}}
      15, f32, f32, tex, vec4i32, 0, 0, 0);
  v = __builtin_amdgcn_image_sample_lz_2d_v4f16_f32( // expected-error {{needs target feature extended-image-insts,16-bit-insts}}
      15, f32, f32, tex, vec4i32, 0, 0, 0);
  v = __builtin_amdgcn_image_sample_lz_2darray_v4f16_f32( // expected-error {{needs target feature extended-image-insts,16-bit-insts}}
      15, f32, f32, f32, tex, vec4i32, 0, 0, 0);
  v = __builtin_amdgcn_image_sample_lz_3d_v4f16_f32( // expected-error {{needs target feature extended-image-insts,16-bit-insts}}
      15, f32, f32, f32, tex, vec4i32, 0, 0, 0);
  v = __builtin_amdgcn_image_sample_lz_cube_v4f16_f32( // expected-error {{needs target feature extended-image-insts,16-bit-insts}}
      15, f32, f32, f32, tex, vec4i32, 0, 0, 0);
  v = __builtin_amdgcn_image_sample_l_1d_v4f16_f32( // expected-error {{needs target feature extended-image-insts,16-bit-insts}}
      15, f32, f32, tex, vec4i32, 0, 0, 0);
  v = __builtin_amdgcn_image_sample_l_1darray_v4f16_f32( // expected-error {{needs target feature extended-image-insts,16-bit-insts}}
      15, f32, f32, f32, tex, vec4i32, 0, 0, 0);
  v = __builtin_amdgcn_image_sample_l_2d_v4f16_f32( // expected-error {{needs target feature extended-image-insts,16-bit-insts}}
      15, f32, f32, f32, tex, vec4i32, 0, 0, 0);
  v = __builtin_amdgcn_image_sample_l_2darray_v4f16_f32( // expected-error {{needs target feature extended-image-insts,16-bit-insts}}
      15, f32, f32, f32, f32, tex, vec4i32, 0, 0, 0);
  v = __builtin_amdgcn_image_sample_l_3d_v4f16_f32( // expected-error {{needs target feature extended-image-insts,16-bit-insts}}
      15, f32, f32, f32, f32, tex, vec4i32, 0, 0, 0);
  v = __builtin_amdgcn_image_sample_l_cube_v4f16_f32( // expected-error {{needs target feature extended-image-insts,16-bit-insts}}
      15, f32, f32, f32, f32, tex, vec4i32, 0, 0, 0);
  v = __builtin_amdgcn_image_sample_d_1d_v4f16_f32( // expected-error {{needs target feature extended-image-insts,16-bit-insts}}
      15, f32, f32, f32, tex, vec4i32, 0, 0, 0);
  v = __builtin_amdgcn_image_sample_d_1darray_v4f16_f32( // expected-error {{needs target feature extended-image-insts,16-bit-insts}}
      15, f32, f32, f32, f32, tex, vec4i32, 0, 0, 0);
  v = __builtin_amdgcn_image_sample_d_2d_v4f16_f32( // expected-error {{needs target feature extended-image-insts,16-bit-insts}}
      15, f32, f32, f32, f32, f32, f32, tex, vec4i32, 0, 0, 0);
  v = __builtin_amdgcn_image_sample_d_2darray_v4f16_f32( // expected-error {{needs target feature extended-image-insts,16-bit-insts}}
      15, f32, f32, f32, f32, f32, f32, f32, tex, vec4i32, 0, 0, 0);
  v = __builtin_amdgcn_image_sample_d_3d_v4f16_f32( // expected-error {{needs target feature extended-image-insts,16-bit-insts}}
      15, f32, f32, f32, f32, f32, f32, f32, f32, f32, tex, vec4i32, 0, 0, 0);
}
