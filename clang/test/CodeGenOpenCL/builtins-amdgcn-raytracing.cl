// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1030 \
// RUN:   -emit-llvm -cl-std=CL2.0 -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1030 -S \
// RUN:   -cl-std=CL2.0 -o - %s | FileCheck -check-prefix=ISA %s

// Test llvm.amdgcn.image.bvh.intersect.ray intrinsic.

// The clang builtin functions __builtin_amdgcn_image_bvh_intersect_ray* use
// postfixes to indicate the types of the 1st, 4th, and 5th arguments.
// By default, the 1st argument is i32, the 4/5-th arguments are float4.
// Postfix l indicates the 1st argument is i64 and postfix h indicates
// the 4/5-th arguments are half4.

typedef unsigned int uint;
typedef unsigned long ulong;
typedef float float4 __attribute__((ext_vector_type(4)));
typedef double double4 __attribute__((ext_vector_type(4)));
typedef half half4 __attribute__((ext_vector_type(4)));
typedef uint uint4 __attribute__((ext_vector_type(4)));

// CHECK: call <4 x i32> @llvm.amdgcn.image.bvh.intersect.ray.i32.v3f32
// ISA: image_bvh_intersect_ray
void test_image_bvh_intersect_ray(global uint4* out, uint node_ptr,
  float ray_extent, float4 ray_origin, float4 ray_dir, float4 ray_inv_dir,
  uint4 texture_descr)
{
  *out = __builtin_amdgcn_image_bvh_intersect_ray(node_ptr, ray_extent,
           ray_origin, ray_dir, ray_inv_dir, texture_descr);
}

// CHECK: call <4 x i32> @llvm.amdgcn.image.bvh.intersect.ray.i32.v3f16
// ISA: image_bvh_intersect_ray
void test_image_bvh_intersect_ray_h(global uint4* out, uint node_ptr,
  float ray_extent, float4 ray_origin, half4 ray_dir, half4 ray_inv_dir,
  uint4 texture_descr)
{
  *out = __builtin_amdgcn_image_bvh_intersect_ray_h(node_ptr, ray_extent,
           ray_origin, ray_dir, ray_inv_dir, texture_descr);
}

// CHECK: call <4 x i32> @llvm.amdgcn.image.bvh.intersect.ray.i64.v3f32
// ISA: image_bvh_intersect_ray
void test_image_bvh_intersect_ray_l(global uint4* out, ulong node_ptr,
  float ray_extent, float4 ray_origin, float4 ray_dir, float4 ray_inv_dir,
  uint4 texture_descr)
{
  *out = __builtin_amdgcn_image_bvh_intersect_ray_l(node_ptr, ray_extent,
           ray_origin, ray_dir, ray_inv_dir, texture_descr);
}

// CHECK: call <4 x i32> @llvm.amdgcn.image.bvh.intersect.ray.i64.v3f16
// ISA: image_bvh_intersect_ray
void test_image_bvh_intersect_ray_lh(global uint4* out, ulong node_ptr,
  float ray_extent, float4 ray_origin, half4 ray_dir, half4 ray_inv_dir,
  uint4 texture_descr)
{
  *out = __builtin_amdgcn_image_bvh_intersect_ray_lh(node_ptr, ray_extent,
           ray_origin, ray_dir, ray_inv_dir, texture_descr);
}

