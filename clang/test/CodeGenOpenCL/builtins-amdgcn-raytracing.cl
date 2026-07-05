// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1030 \
// RUN:   -emit-llvm -cl-std=CL2.0 -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1030 -S \
// RUN:   -cl-std=CL2.0 -o - %s | FileCheck -check-prefix=ISA %s
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1200 -emit-llvm \
// RUN:   -cl-std=CL2.0 -o - %s | FileCheck -check-prefix=GFX12 %s
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1200 -S \
// RUN:   -cl-std=CL2.0 -o - %s | FileCheck -check-prefix=GFX12ISA %s

// Test llvm.amdgcn.image.bvh.intersect.ray intrinsic.

// The clang builtin functions __builtin_amdgcn_image_bvh_intersect_ray* use
// postfixes to indicate the types of the 1st, 4th, and 5th arguments.
// By default, the 1st argument is i32, the 4/5-th arguments are float4.
// Postfix l indicates the 1st argument is i64 and postfix h indicates
// the 4/5-th arguments are half4.

typedef unsigned char uchar;
typedef unsigned int uint;
typedef unsigned long ulong;
typedef float float3 __attribute__((ext_vector_type(3)));
typedef float float4 __attribute__((ext_vector_type(4)));
typedef double double4 __attribute__((ext_vector_type(4)));
typedef half half4 __attribute__((ext_vector_type(4)));
typedef uint uint2 __attribute__((ext_vector_type(2)));
typedef uint uint4 __attribute__((ext_vector_type(4)));
typedef uint uint8 __attribute__((ext_vector_type(8)));
typedef uint uint10 __attribute__((ext_vector_type(10)));
typedef ulong ulong2 __attribute__((ext_vector_type(2)));

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

#if __has_builtin(__builtin_amdgcn_image_bvh8_intersect_ray)
// GFX12: call { <10 x i32>, <3 x float>, <3 x float> } @llvm.amdgcn.image.bvh8.intersect.ray(
// GFX12: i64 %node_ptr, float %ray_extent, i8 %instance_mask, <3 x float> %ray_origin,
// GFX12: <3 x float> %ray_dir, i32 %offset, <4 x i32> %texture_descr)
// GFX12ISA: image_bvh8_intersect_ray
void test_image_bvh8_intersect_ray(global uint10* ret_vdata, float3* ret_ray_origin,
    float3* ret_ray_dir, ulong node_ptr, float ray_extent, uchar instance_mask,
    float3 ray_origin, float3 ray_dir, uint offset, uint4 texture_descr)
{
  *ret_vdata = __builtin_amdgcn_image_bvh8_intersect_ray(node_ptr, ray_extent,
           instance_mask, ray_origin, ray_dir, offset, texture_descr,
           ret_ray_origin, ret_ray_dir);
}
#endif

#if __has_builtin(__builtin_amdgcn_image_bvh_dual_intersect_ray)
// GFX12: call { <10 x i32>, <3 x float>, <3 x float> } @llvm.amdgcn.image.bvh.dual.intersect.ray(
// GFX12: i64 %node_ptr, float %ray_extent, i8 %instance_mask, <3 x float> %ray_origin,
// GFX12: <3 x float> %ray_dir, <2 x i32> %offset, <4 x i32> %texture_descr)
// GFX12ISA: image_bvh_dual_intersect_ray
void test_builtin_amdgcn_image_bvh_dual_intersect_ray(global uint10* ret_vdata, float3* ret_ray_origin,
    float3* ret_ray_dir, ulong node_ptr, float ray_extent, uchar instance_mask,
    float3 ray_origin, float3 ray_dir, uint2 offset, uint4 texture_descr)
{
  *ret_vdata = __builtin_amdgcn_image_bvh_dual_intersect_ray(node_ptr, ray_extent,
           instance_mask, ray_origin, ray_dir, offset, texture_descr,
           ret_ray_origin, ret_ray_dir);
}
#endif

#if __has_builtin(__builtin_amdgcn_ds_bvh_stack_push4_pop1_rtn)
// GFX12: call { i32, i32 } @llvm.amdgcn.ds.bvh.stack.push4.pop1.rtn(
// GFX12: i32 %addr, i32 %data0, <4 x i32> %data1, i32 0)
// GFX12ISA: ds_bvh_stack_push4_pop1_rtn
void test_builtin_amdgcn_ds_bvh_stack_push4_pop1_rtn(uint* ret_vdst, uint* ret_addr,
    uint addr, uint data0, uint4 data1)
{
  uint2 ret = __builtin_amdgcn_ds_bvh_stack_push4_pop1_rtn(addr, data0, data1, /*constant offset=*/0);
  *ret_vdst = ret.x;
  *ret_addr = ret.y;
}
#endif

#if __has_builtin(__builtin_amdgcn_ds_bvh_stack_push8_pop1_rtn)
// GFX12: call { i32, i32 } @llvm.amdgcn.ds.bvh.stack.push8.pop1.rtn(
// GFX12: i32 %addr, i32 %data0, <8 x i32> %data1, i32 0)
// GFX12ISA: ds_bvh_stack_push8_pop1_rtn
void test_builtin_amdgcn_ds_bvh_stack_push8_pop1_rtn(uint* ret_vdst, uint* ret_addr,
    uint addr, uint data0, uint8 data1)
{
  uint2 ret = __builtin_amdgcn_ds_bvh_stack_push8_pop1_rtn(addr, data0, data1, /*constant offset=*/0);
  *ret_vdst = ret.x;
  *ret_addr = ret.y;
}
#endif

#if __has_builtin(__builtin_amdgcn_ds_bvh_stack_push8_pop2_rtn)
// GFX12: call { i64, i32 } @llvm.amdgcn.ds.bvh.stack.push8.pop2.rtn(
// GFX12: i32 %addr, i32 %data0, <8 x i32> %data1, i32 0)
// GFX12ISA: ds_bvh_stack_push8_pop2_rtn
void test_builtin_amdgcn_ds_bvh_stack_push8_pop2_rtn(ulong* ret_vdst, uint* ret_addr,
    uint addr, uint data0, uint8 data1)
{
  ulong2 ret = __builtin_amdgcn_ds_bvh_stack_push8_pop2_rtn(addr, data0, data1, /*constant offset=*/0);
  *ret_vdst = ret.x;
  *ret_addr = ret.y;
}
#endif
