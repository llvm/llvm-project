// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgpu9.42 -fcuda-is-device -emit-llvm -verify=device -o - %s
// RUN: %clang_cc1 -triple x86_64 -aux-triple amdgpu -fsyntax-only -verify=host %s

// host-no-diagnostics

#define __device__ __attribute__((device))

typedef float float4 __attribute__((ext_vector_type(4)));

__device__ float4 test_image(float4 v4f32, int i32, __amdgpu_texture_t tex) {
  __builtin_amdgcn_image_store_2d_v4f32_i32(v4f32, 15, i32, i32, tex, 0, 0); // device-error {{'__builtin_amdgcn_image_store_2d_v4f32_i32' needs target feature image-insts}}
  return __builtin_amdgcn_image_load_2d_v4f32_i32(15, i32, i32, tex, 0, 0); // device-error {{'__builtin_amdgcn_image_load_2d_v4f32_i32' needs target feature image-insts}}
}
