// RUN: %clang_cc1 -triple amdgpu9.42-- -emit-llvm -verify -o - %s
// REQUIRES: amdgpu-registered-target

// expected-no-diagnostics

typedef int int4 __attribute__((ext_vector_type(4)));
typedef float float4 __attribute__((ext_vector_type(4)));

void test_guarded(float4 v4f32, float f32, int i32, __amdgpu_texture_t tex, int4 vec4i32) {
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_image_load_2d_v4f32_i32))
    v4f32 = __builtin_amdgcn_image_load_2d_v4f32_i32(15, i32, i32, tex, 0, 0);
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_image_store_2d_v4f32_i32))
    __builtin_amdgcn_image_store_2d_v4f32_i32(v4f32, 15, i32, i32, tex, 0, 0);
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_image_sample_lz_2d_v4f32_f32))
    v4f32 = __builtin_amdgcn_image_sample_lz_2d_v4f32_f32(15, f32, f32, tex, vec4i32, 0, 0, 0);
}
