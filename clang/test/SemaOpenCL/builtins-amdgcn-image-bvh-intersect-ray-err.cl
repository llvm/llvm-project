// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgpu12.5 -verify -S -o - %s
// RUN: %clang_cc1 -triple amdgpu12.50 -verify -S -o - %s
// RUN: %clang_cc1 -triple amdgpu12.51 -verify -S -o - %s

typedef unsigned int uint4 __attribute__((ext_vector_type(4)));
typedef float float4 __attribute__((ext_vector_type(4)));

// gfx12.5 does not have the BVH ray tracing instructions; unlike gfx1200 it does
// not inherit bvh-ray-tracing-insts from the GFX12 generation.
void test_image_bvh_intersect_ray(global uint4 *out, unsigned node, float ext,
                                  float4 origin, float4 dir, float4 invdir,
                                  uint4 desc) {
  *out = __builtin_amdgcn_image_bvh_intersect_ray(node, ext, origin, dir, invdir, desc); // expected-error{{'__builtin_amdgcn_image_bvh_intersect_ray' needs target feature bvh-ray-tracing-insts}}
}
