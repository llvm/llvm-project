#include "../CodeGenCUDA/Inputs/cuda.h"

// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -target-cpu gfx1100 -target-feature +extended-image-insts \
// RUN:            -fcuda-is-device -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -target-cpu gfx1100 -target-feature +extended-image-insts \
// RUN:            -fcuda-is-device -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 \
// RUN:            -target-cpu gfx1100 -target-feature +extended-image-insts \
// RUN:            -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

typedef float float4 __attribute__((ext_vector_type(4)));
typedef int int4 __attribute__((ext_vector_type(4)));
typedef _Float16 half;
typedef half half4 __attribute__((ext_vector_type(4)));

// CIR-LABEL: @_Z26test_gather4_lz_2d_v4f32_r
// CIR: cir.call_llvm_intrinsic "amdgcn.image.gather4.lz.2d" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.vector<8 x !s32i>, !cir.vector<4 x !s32i>, !cir.bool, !s32i, !s32i) -> !cir.vector<4 x !cir.float>
// LLVM: define{{.*}} <4 x float> @_Z26test_gather4_lz_2d_v4f32_rffu18__amdgpu_texture_tDv4_i(
// LLVM: call {{.*}}<4 x float> @llvm.amdgcn.image.gather4.lz.2d.v4f32.f32.v8i32.v4i32(i32 1, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ float4 test_gather4_lz_2d_v4f32_r(float s, float t, __amdgpu_texture_t tex, int4 samp) {
  return __builtin_amdgcn_image_gather4_lz_2d_v4f32_f32(1, s, t, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @_Z26test_gather4_lz_2d_v4f32_g
// CIR: cir.call_llvm_intrinsic "amdgcn.image.gather4.lz.2d" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.vector<8 x !s32i>, !cir.vector<4 x !s32i>, !cir.bool, !s32i, !s32i) -> !cir.vector<4 x !cir.float>
// LLVM: define{{.*}} <4 x float> @_Z26test_gather4_lz_2d_v4f32_gffu18__amdgpu_texture_tDv4_i(
// LLVM: call {{.*}}<4 x float> @llvm.amdgcn.image.gather4.lz.2d.v4f32.f32.v8i32.v4i32(i32 2, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ float4 test_gather4_lz_2d_v4f32_g(float s, float t, __amdgpu_texture_t tex, int4 samp) {
  return __builtin_amdgcn_image_gather4_lz_2d_v4f32_f32(2, s, t, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @_Z26test_gather4_lz_2d_v4f32_b
// CIR: cir.call_llvm_intrinsic "amdgcn.image.gather4.lz.2d" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.vector<8 x !s32i>, !cir.vector<4 x !s32i>, !cir.bool, !s32i, !s32i) -> !cir.vector<4 x !cir.float>
// LLVM: define{{.*}} <4 x float> @_Z26test_gather4_lz_2d_v4f32_bffu18__amdgpu_texture_tDv4_i(
// LLVM: call {{.*}}<4 x float> @llvm.amdgcn.image.gather4.lz.2d.v4f32.f32.v8i32.v4i32(i32 4, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ float4 test_gather4_lz_2d_v4f32_b(float s, float t, __amdgpu_texture_t tex, int4 samp) {
  return __builtin_amdgcn_image_gather4_lz_2d_v4f32_f32(4, s, t, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @_Z26test_gather4_lz_2d_v4f32_a
// CIR: cir.call_llvm_intrinsic "amdgcn.image.gather4.lz.2d" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.vector<8 x !s32i>, !cir.vector<4 x !s32i>, !cir.bool, !s32i, !s32i) -> !cir.vector<4 x !cir.float>
// LLVM: define{{.*}} <4 x float> @_Z26test_gather4_lz_2d_v4f32_affu18__amdgpu_texture_tDv4_i(
// LLVM: call {{.*}}<4 x float> @llvm.amdgcn.image.gather4.lz.2d.v4f32.f32.v8i32.v4i32(i32 8, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ float4 test_gather4_lz_2d_v4f32_a(float s, float t, __amdgpu_texture_t tex, int4 samp) {
  return __builtin_amdgcn_image_gather4_lz_2d_v4f32_f32(8, s, t, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @_Z23test_sample_lz_1d_v4f32
// CIR: cir.call_llvm_intrinsic "amdgcn.image.sample.lz.1d" {{.*}} : (!s32i, !cir.float, !cir.vector<8 x !s32i>, !cir.vector<4 x !s32i>, !cir.bool, !s32i, !s32i) -> !cir.vector<4 x !cir.float>
// LLVM: define{{.*}} <4 x float> @_Z23test_sample_lz_1d_v4f32fu18__amdgpu_texture_tDv4_i(
// LLVM: call {{.*}}<4 x float> @llvm.amdgcn.image.sample.lz.1d.v4f32.f32.v8i32.v4i32(i32 {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ float4 test_sample_lz_1d_v4f32(float s, __amdgpu_texture_t tex, int4 samp) {
  return __builtin_amdgcn_image_sample_lz_1d_v4f32_f32(15, s, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @_Z22test_sample_l_1d_v4f32
// CIR: cir.call_llvm_intrinsic "amdgcn.image.sample.l.1d" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.vector<8 x !s32i>, !cir.vector<4 x !s32i>, !cir.bool, !s32i, !s32i) -> !cir.vector<4 x !cir.float>
// LLVM: define{{.*}} <4 x float> @_Z22test_sample_l_1d_v4f32ffu18__amdgpu_texture_tDv4_i(
// LLVM: call {{.*}}<4 x float> @llvm.amdgcn.image.sample.l.1d.v4f32.f32.v8i32.v4i32(i32 {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ float4 test_sample_l_1d_v4f32(float s, float lod, __amdgpu_texture_t tex, int4 samp) {
  return __builtin_amdgcn_image_sample_l_1d_v4f32_f32(15, s, lod, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @_Z22test_sample_d_1d_v4f32
// CIR: cir.call_llvm_intrinsic "amdgcn.image.sample.d.1d" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.float, !cir.vector<8 x !s32i>, !cir.vector<4 x !s32i>, !cir.bool, !s32i, !s32i) -> !cir.vector<4 x !cir.float>
// LLVM: define{{.*}} <4 x float> @_Z22test_sample_d_1d_v4f32fffu18__amdgpu_texture_tDv4_i(
// LLVM: call {{.*}}<4 x float> @llvm.amdgcn.image.sample.d.1d.v4f32.f32.f32.v8i32.v4i32(i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ float4 test_sample_d_1d_v4f32(float dsdx, float dsdy, float s, __amdgpu_texture_t tex, int4 samp) {
  return __builtin_amdgcn_image_sample_d_1d_v4f32_f32(15, dsdx, dsdy, s, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @_Z23test_sample_lz_2d_v4f32
// CIR: cir.call_llvm_intrinsic "amdgcn.image.sample.lz.2d" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.vector<8 x !s32i>, !cir.vector<4 x !s32i>, !cir.bool, !s32i, !s32i) -> !cir.vector<4 x !cir.float>
// LLVM: define{{.*}} <4 x float> @_Z23test_sample_lz_2d_v4f32ffu18__amdgpu_texture_tDv4_i(
// LLVM: call {{.*}}<4 x float> @llvm.amdgcn.image.sample.lz.2d.v4f32.f32.v8i32.v4i32(i32 {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ float4 test_sample_lz_2d_v4f32(float s, float t, __amdgpu_texture_t tex, int4 samp) {
  return __builtin_amdgcn_image_sample_lz_2d_v4f32_f32(15, s, t, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @_Z22test_sample_l_2d_v4f32
// CIR: cir.call_llvm_intrinsic "amdgcn.image.sample.l.2d" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.float, !cir.vector<8 x !s32i>, !cir.vector<4 x !s32i>, !cir.bool, !s32i, !s32i) -> !cir.vector<4 x !cir.float>
// LLVM: define{{.*}} <4 x float> @_Z22test_sample_l_2d_v4f32fffu18__amdgpu_texture_tDv4_i(
// LLVM: call {{.*}}<4 x float> @llvm.amdgcn.image.sample.l.2d.v4f32.f32.v8i32.v4i32(i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ float4 test_sample_l_2d_v4f32(float s, float t, float lod, __amdgpu_texture_t tex, int4 samp) {
  return __builtin_amdgcn_image_sample_l_2d_v4f32_f32(10, s, t, lod, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @_Z22test_sample_d_2d_v4f32
// CIR: cir.call_llvm_intrinsic "amdgcn.image.sample.d.2d" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.float, !cir.float, !cir.float, !cir.float, !cir.vector<8 x !s32i>, !cir.vector<4 x !s32i>, !cir.bool, !s32i, !s32i) -> !cir.vector<4 x !cir.float>
// LLVM: define{{.*}} <4 x float> @_Z22test_sample_d_2d_v4f32ffffffu18__amdgpu_texture_tDv4_i(
// LLVM: call {{.*}}<4 x float> @llvm.amdgcn.image.sample.d.2d.v4f32.f32.f32.v8i32.v4i32(i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ float4 test_sample_d_2d_v4f32(float dsdx, float dtdx, float dsdy, float dtdy, float s, float t, __amdgpu_texture_t tex, int4 samp) {
  return __builtin_amdgcn_image_sample_d_2d_v4f32_f32(15, dsdx, dtdx, dsdy, dtdy, s, t, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @_Z23test_sample_lz_3d_v4f32
// CIR: cir.call_llvm_intrinsic "amdgcn.image.sample.lz.3d" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.float, !cir.vector<8 x !s32i>, !cir.vector<4 x !s32i>, !cir.bool, !s32i, !s32i) -> !cir.vector<4 x !cir.float>
// LLVM: define{{.*}} <4 x float> @_Z23test_sample_lz_3d_v4f32fffu18__amdgpu_texture_tDv4_i(
// LLVM: call {{.*}}<4 x float> @llvm.amdgcn.image.sample.lz.3d.v4f32.f32.v8i32.v4i32(i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ float4 test_sample_lz_3d_v4f32(float s, float t, float r, __amdgpu_texture_t tex, int4 samp) {
  return __builtin_amdgcn_image_sample_lz_3d_v4f32_f32(15, s, t, r, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @_Z22test_sample_l_3d_v4f32
// CIR: cir.call_llvm_intrinsic "amdgcn.image.sample.l.3d" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.float, !cir.float, !cir.vector<8 x !s32i>, !cir.vector<4 x !s32i>, !cir.bool, !s32i, !s32i) -> !cir.vector<4 x !cir.float>
// LLVM: define{{.*}} <4 x float> @_Z22test_sample_l_3d_v4f32ffffu18__amdgpu_texture_tDv4_i(
// LLVM: call {{.*}}<4 x float> @llvm.amdgcn.image.sample.l.3d.v4f32.f32.v8i32.v4i32(i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ float4 test_sample_l_3d_v4f32(float s, float t, float r, float lod, __amdgpu_texture_t tex, int4 samp) {
  return __builtin_amdgcn_image_sample_l_3d_v4f32_f32(1, s, t, r, lod, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @_Z22test_sample_d_3d_v4f32
// CIR: cir.call_llvm_intrinsic "amdgcn.image.sample.d.3d" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.float, !cir.float, !cir.float, !cir.float, !cir.float, !cir.float, !cir.float, !cir.vector<8 x !s32i>, !cir.vector<4 x !s32i>, !cir.bool, !s32i, !s32i) -> !cir.vector<4 x !cir.float>
// LLVM: define{{.*}} <4 x float> @_Z22test_sample_d_3d_v4f32fffffffffu18__amdgpu_texture_tDv4_i(
// LLVM: call {{.*}}<4 x float> @llvm.amdgcn.image.sample.d.3d.v4f32.f32.f32.v8i32.v4i32(i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ float4 test_sample_d_3d_v4f32(float dsdx, float dtdx, float drdx, float dsdy, float dtdy, float drdy, float s, float t, float r, __amdgpu_texture_t tex, int4 samp) {
  return __builtin_amdgcn_image_sample_d_3d_v4f32_f32(1, dsdx, dtdx, drdx, dsdy, dtdy, drdy, s, t, r, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @_Z25test_sample_lz_cube_v4f32
// CIR: cir.call_llvm_intrinsic "amdgcn.image.sample.lz.cube" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.float, !cir.vector<8 x !s32i>, !cir.vector<4 x !s32i>, !cir.bool, !s32i, !s32i) -> !cir.vector<4 x !cir.float>
// LLVM: define{{.*}} <4 x float> @_Z25test_sample_lz_cube_v4f32fffu18__amdgpu_texture_tDv4_i(
// LLVM: call {{.*}}<4 x float> @llvm.amdgcn.image.sample.lz.cube.v4f32.f32.v8i32.v4i32(i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ float4 test_sample_lz_cube_v4f32(float s, float t, float face, __amdgpu_texture_t tex, int4 samp) {
  return __builtin_amdgcn_image_sample_lz_cube_v4f32_f32(1, s, t, face, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @_Z24test_sample_l_cube_v4f32
// CIR: cir.call_llvm_intrinsic "amdgcn.image.sample.l.cube" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.float, !cir.float, !cir.vector<8 x !s32i>, !cir.vector<4 x !s32i>, !cir.bool, !s32i, !s32i) -> !cir.vector<4 x !cir.float>
// LLVM: define{{.*}} <4 x float> @_Z24test_sample_l_cube_v4f32ffffu18__amdgpu_texture_tDv4_i(
// LLVM: call {{.*}}<4 x float> @llvm.amdgcn.image.sample.l.cube.v4f32.f32.v8i32.v4i32(i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ float4 test_sample_l_cube_v4f32(float s, float t, float face, float lod, __amdgpu_texture_t tex, int4 samp) {
  return __builtin_amdgcn_image_sample_l_cube_v4f32_f32(1, s, t, face, lod, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @_Z28test_sample_lz_1darray_v4f32
// CIR: cir.call_llvm_intrinsic "amdgcn.image.sample.lz.1darray" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.vector<8 x !s32i>, !cir.vector<4 x !s32i>, !cir.bool, !s32i, !s32i) -> !cir.vector<4 x !cir.float>
// LLVM: define{{.*}} <4 x float> @_Z28test_sample_lz_1darray_v4f32ffu18__amdgpu_texture_tDv4_i(
// LLVM: call {{.*}}<4 x float> @llvm.amdgcn.image.sample.lz.1darray.v4f32.f32.v8i32.v4i32(i32 {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ float4 test_sample_lz_1darray_v4f32(float s, float slice, __amdgpu_texture_t tex, int4 samp) {
  return __builtin_amdgcn_image_sample_lz_1darray_v4f32_f32(1, s, slice, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @_Z27test_sample_l_1darray_v4f32
// CIR: cir.call_llvm_intrinsic "amdgcn.image.sample.l.1darray" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.float, !cir.vector<8 x !s32i>, !cir.vector<4 x !s32i>, !cir.bool, !s32i, !s32i) -> !cir.vector<4 x !cir.float>
// LLVM: define{{.*}} <4 x float> @_Z27test_sample_l_1darray_v4f32fffu18__amdgpu_texture_tDv4_i(
// LLVM: call {{.*}}<4 x float> @llvm.amdgcn.image.sample.l.1darray.v4f32.f32.v8i32.v4i32(i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ float4 test_sample_l_1darray_v4f32(float s, float slice, float lod, __amdgpu_texture_t tex, int4 samp) {
  return __builtin_amdgcn_image_sample_l_1darray_v4f32_f32(1, s, slice, lod, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @_Z27test_sample_d_1darray_v4f32
// CIR: cir.call_llvm_intrinsic "amdgcn.image.sample.d.1darray" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.float, !cir.float, !cir.vector<8 x !s32i>, !cir.vector<4 x !s32i>, !cir.bool, !s32i, !s32i) -> !cir.vector<4 x !cir.float>
// LLVM: define{{.*}} <4 x float> @_Z27test_sample_d_1darray_v4f32ffffu18__amdgpu_texture_tDv4_i(
// LLVM: call {{.*}}<4 x float> @llvm.amdgcn.image.sample.d.1darray.v4f32.f32.f32.v8i32.v4i32(i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ float4 test_sample_d_1darray_v4f32(float dsdx, float dsdy, float s, float slice, __amdgpu_texture_t tex, int4 samp) {
  return __builtin_amdgcn_image_sample_d_1darray_v4f32_f32(1, dsdx, dsdy, s, slice, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @_Z28test_sample_lz_2darray_v4f32
// CIR: cir.call_llvm_intrinsic "amdgcn.image.sample.lz.2darray" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.float, !cir.vector<8 x !s32i>, !cir.vector<4 x !s32i>, !cir.bool, !s32i, !s32i) -> !cir.vector<4 x !cir.float>
// LLVM: define{{.*}} <4 x float> @_Z28test_sample_lz_2darray_v4f32fffu18__amdgpu_texture_tDv4_i(
// LLVM: call {{.*}}<4 x float> @llvm.amdgcn.image.sample.lz.2darray.v4f32.f32.v8i32.v4i32(i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ float4 test_sample_lz_2darray_v4f32(float s, float t, float slice, __amdgpu_texture_t tex, int4 samp) {
  return __builtin_amdgcn_image_sample_lz_2darray_v4f32_f32(1, s, t, slice, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @_Z27test_sample_l_2darray_v4f32
// CIR: cir.call_llvm_intrinsic "amdgcn.image.sample.l.2darray" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.float, !cir.float, !cir.vector<8 x !s32i>, !cir.vector<4 x !s32i>, !cir.bool, !s32i, !s32i) -> !cir.vector<4 x !cir.float>
// LLVM: define{{.*}} <4 x float> @_Z27test_sample_l_2darray_v4f32ffffu18__amdgpu_texture_tDv4_i(
// LLVM: call {{.*}}<4 x float> @llvm.amdgcn.image.sample.l.2darray.v4f32.f32.v8i32.v4i32(i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ float4 test_sample_l_2darray_v4f32(float s, float t, float slice, float lod, __amdgpu_texture_t tex, int4 samp) {
  return __builtin_amdgcn_image_sample_l_2darray_v4f32_f32(1, s, t, slice, lod, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @_Z27test_sample_d_2darray_v4f32
// CIR: cir.call_llvm_intrinsic "amdgcn.image.sample.d.2darray" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.float, !cir.float, !cir.float, !cir.float, !cir.float, !cir.vector<8 x !s32i>, !cir.vector<4 x !s32i>, !cir.bool, !s32i, !s32i) -> !cir.vector<4 x !cir.float>
// LLVM: define{{.*}} <4 x float> @_Z27test_sample_d_2darray_v4f32fffffffu18__amdgpu_texture_tDv4_i(
// LLVM: call {{.*}}<4 x float> @llvm.amdgcn.image.sample.d.2darray.v4f32.f32.f32.v8i32.v4i32(i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ float4 test_sample_d_2darray_v4f32(float dsdx, float dtdx, float dsdy, float dtdy, float s, float t, float slice, __amdgpu_texture_t tex, int4 samp) {
  return __builtin_amdgcn_image_sample_d_2darray_v4f32_f32(1, dsdx, dtdx, dsdy, dtdy, s, t, slice, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @_Z23test_sample_lz_1d_v4f16
// CIR: cir.call_llvm_intrinsic "amdgcn.image.sample.lz.1d" {{.*}} : (!s32i, !cir.float, !cir.vector<8 x !s32i>, !cir.vector<4 x !s32i>, !cir.bool, !s32i, !s32i) -> !cir.vector<4 x !cir.f16>
// LLVM: define{{.*}} <4 x half> @_Z23test_sample_lz_1d_v4f16fu18__amdgpu_texture_tDv4_i(
// LLVM: call {{.*}}<4 x half> @llvm.amdgcn.image.sample.lz.1d.v4f16.f32.v8i32.v4i32(i32 {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ half4 test_sample_lz_1d_v4f16(float s, __amdgpu_texture_t tex, int4 samp) {
  return __builtin_amdgcn_image_sample_lz_1d_v4f16_f32(15, s, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @_Z22test_sample_l_1d_v4f16
// CIR: cir.call_llvm_intrinsic "amdgcn.image.sample.l.1d" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.vector<8 x !s32i>, !cir.vector<4 x !s32i>, !cir.bool, !s32i, !s32i) -> !cir.vector<4 x !cir.f16>
// LLVM: define{{.*}} <4 x half> @_Z22test_sample_l_1d_v4f16ffu18__amdgpu_texture_tDv4_i(
// LLVM: call {{.*}}<4 x half> @llvm.amdgcn.image.sample.l.1d.v4f16.f32.v8i32.v4i32(i32 {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ half4 test_sample_l_1d_v4f16(float s, float lod, __amdgpu_texture_t tex, int4 samp) {
  return __builtin_amdgcn_image_sample_l_1d_v4f16_f32(15, s, lod, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @_Z22test_sample_d_1d_v4f16
// CIR: cir.call_llvm_intrinsic "amdgcn.image.sample.d.1d" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.float, !cir.vector<8 x !s32i>, !cir.vector<4 x !s32i>, !cir.bool, !s32i, !s32i) -> !cir.vector<4 x !cir.f16>
// LLVM: define{{.*}} <4 x half> @_Z22test_sample_d_1d_v4f16fffu18__amdgpu_texture_tDv4_i(
// LLVM: call {{.*}}<4 x half> @llvm.amdgcn.image.sample.d.1d.v4f16.f32.f32.v8i32.v4i32(i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ half4 test_sample_d_1d_v4f16(float dsdx, float dsdy, float s, __amdgpu_texture_t tex, int4 samp) {
  return __builtin_amdgcn_image_sample_d_1d_v4f16_f32(15, dsdx, dsdy, s, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @_Z23test_sample_lz_2d_v4f16
// CIR: cir.call_llvm_intrinsic "amdgcn.image.sample.lz.2d" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.vector<8 x !s32i>, !cir.vector<4 x !s32i>, !cir.bool, !s32i, !s32i) -> !cir.vector<4 x !cir.f16>
// LLVM: define{{.*}} <4 x half> @_Z23test_sample_lz_2d_v4f16ffu18__amdgpu_texture_tDv4_i(
// LLVM: call {{.*}}<4 x half> @llvm.amdgcn.image.sample.lz.2d.v4f16.f32.v8i32.v4i32(i32 {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ half4 test_sample_lz_2d_v4f16(float s, float t, __amdgpu_texture_t tex, int4 samp) {
  return __builtin_amdgcn_image_sample_lz_2d_v4f16_f32(15, s, t, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @_Z22test_sample_l_2d_v4f16
// CIR: cir.call_llvm_intrinsic "amdgcn.image.sample.l.2d" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.float, !cir.vector<8 x !s32i>, !cir.vector<4 x !s32i>, !cir.bool, !s32i, !s32i) -> !cir.vector<4 x !cir.f16>
// LLVM: define{{.*}} <4 x half> @_Z22test_sample_l_2d_v4f16fffu18__amdgpu_texture_tDv4_i(
// LLVM: call {{.*}}<4 x half> @llvm.amdgcn.image.sample.l.2d.v4f16.f32.v8i32.v4i32(i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ half4 test_sample_l_2d_v4f16(float s, float t, float lod, __amdgpu_texture_t tex, int4 samp) {
  return __builtin_amdgcn_image_sample_l_2d_v4f16_f32(15, s, t, lod, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @_Z22test_sample_d_2d_v4f16
// CIR: cir.call_llvm_intrinsic "amdgcn.image.sample.d.2d" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.float, !cir.float, !cir.float, !cir.float, !cir.vector<8 x !s32i>, !cir.vector<4 x !s32i>, !cir.bool, !s32i, !s32i) -> !cir.vector<4 x !cir.f16>
// LLVM: define{{.*}} <4 x half> @_Z22test_sample_d_2d_v4f16ffffffu18__amdgpu_texture_tDv4_i(
// LLVM: call {{.*}}<4 x half> @llvm.amdgcn.image.sample.d.2d.v4f16.f32.f32.v8i32.v4i32(i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ half4 test_sample_d_2d_v4f16(float dsdx, float dtdx, float dsdy, float dtdy, float s, float t, __amdgpu_texture_t tex, int4 samp) {
  return __builtin_amdgcn_image_sample_d_2d_v4f16_f32(15, dsdx, dtdx, dsdy, dtdy, s, t, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @_Z23test_sample_lz_3d_v4f16
// CIR: cir.call_llvm_intrinsic "amdgcn.image.sample.lz.3d" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.float, !cir.vector<8 x !s32i>, !cir.vector<4 x !s32i>, !cir.bool, !s32i, !s32i) -> !cir.vector<4 x !cir.f16>
// LLVM: define{{.*}} <4 x half> @_Z23test_sample_lz_3d_v4f16fffu18__amdgpu_texture_tDv4_i(
// LLVM: call {{.*}}<4 x half> @llvm.amdgcn.image.sample.lz.3d.v4f16.f32.v8i32.v4i32(i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ half4 test_sample_lz_3d_v4f16(float s, float t, float r, __amdgpu_texture_t tex, int4 samp) {
  return __builtin_amdgcn_image_sample_lz_3d_v4f16_f32(15, s, t, r, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @_Z22test_sample_l_3d_v4f16
// CIR: cir.call_llvm_intrinsic "amdgcn.image.sample.l.3d" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.float, !cir.float, !cir.vector<8 x !s32i>, !cir.vector<4 x !s32i>, !cir.bool, !s32i, !s32i) -> !cir.vector<4 x !cir.f16>
// LLVM: define{{.*}} <4 x half> @_Z22test_sample_l_3d_v4f16ffffu18__amdgpu_texture_tDv4_i(
// LLVM: call {{.*}}<4 x half> @llvm.amdgcn.image.sample.l.3d.v4f16.f32.v8i32.v4i32(i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ half4 test_sample_l_3d_v4f16(float s, float t, float r, float lod, __amdgpu_texture_t tex, int4 samp) {
  return __builtin_amdgcn_image_sample_l_3d_v4f16_f32(15, s, t, r, lod, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @_Z22test_sample_d_3d_v4f16
// CIR: cir.call_llvm_intrinsic "amdgcn.image.sample.d.3d" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.float, !cir.float, !cir.float, !cir.float, !cir.float, !cir.float, !cir.float, !cir.vector<8 x !s32i>, !cir.vector<4 x !s32i>, !cir.bool, !s32i, !s32i) -> !cir.vector<4 x !cir.f16>
// LLVM: define{{.*}} <4 x half> @_Z22test_sample_d_3d_v4f16fffffffffu18__amdgpu_texture_tDv4_i(
// LLVM: call {{.*}}<4 x half> @llvm.amdgcn.image.sample.d.3d.v4f16.f32.f32.v8i32.v4i32(i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ half4 test_sample_d_3d_v4f16(float dsdx, float dtdx, float drdx, float dsdy, float dtdy, float drdy, float s, float t, float r, __amdgpu_texture_t tex, int4 samp) {
  return __builtin_amdgcn_image_sample_d_3d_v4f16_f32(15, dsdx, dtdx, drdx, dsdy, dtdy, drdy, s, t, r, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @_Z25test_sample_lz_cube_v4f16
// CIR: cir.call_llvm_intrinsic "amdgcn.image.sample.lz.cube" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.float, !cir.vector<8 x !s32i>, !cir.vector<4 x !s32i>, !cir.bool, !s32i, !s32i) -> !cir.vector<4 x !cir.f16>
// LLVM: define{{.*}} <4 x half> @_Z25test_sample_lz_cube_v4f16fffu18__amdgpu_texture_tDv4_i(
// LLVM: call {{.*}}<4 x half> @llvm.amdgcn.image.sample.lz.cube.v4f16.f32.v8i32.v4i32(i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ half4 test_sample_lz_cube_v4f16(float s, float t, float face, __amdgpu_texture_t tex, int4 samp) {
  return __builtin_amdgcn_image_sample_lz_cube_v4f16_f32(15, s, t, face, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @_Z24test_sample_l_cube_v4f16
// CIR: cir.call_llvm_intrinsic "amdgcn.image.sample.l.cube" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.float, !cir.float, !cir.vector<8 x !s32i>, !cir.vector<4 x !s32i>, !cir.bool, !s32i, !s32i) -> !cir.vector<4 x !cir.f16>
// LLVM: define{{.*}} <4 x half> @_Z24test_sample_l_cube_v4f16ffffu18__amdgpu_texture_tDv4_i(
// LLVM: call {{.*}}<4 x half> @llvm.amdgcn.image.sample.l.cube.v4f16.f32.v8i32.v4i32(i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ half4 test_sample_l_cube_v4f16(float s, float t, float face, float lod, __amdgpu_texture_t tex, int4 samp) {
  return __builtin_amdgcn_image_sample_l_cube_v4f16_f32(15, s, t, face, lod, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @_Z28test_sample_lz_1darray_v4f16
// CIR: cir.call_llvm_intrinsic "amdgcn.image.sample.lz.1darray" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.vector<8 x !s32i>, !cir.vector<4 x !s32i>, !cir.bool, !s32i, !s32i) -> !cir.vector<4 x !cir.f16>
// LLVM: define{{.*}} <4 x half> @_Z28test_sample_lz_1darray_v4f16ffu18__amdgpu_texture_tDv4_i(
// LLVM: call {{.*}}<4 x half> @llvm.amdgcn.image.sample.lz.1darray.v4f16.f32.v8i32.v4i32(i32 {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ half4 test_sample_lz_1darray_v4f16(float s, float slice, __amdgpu_texture_t tex, int4 samp) {
  return __builtin_amdgcn_image_sample_lz_1darray_v4f16_f32(15, s, slice, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @_Z27test_sample_l_1darray_v4f16
// CIR: cir.call_llvm_intrinsic "amdgcn.image.sample.l.1darray" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.float, !cir.vector<8 x !s32i>, !cir.vector<4 x !s32i>, !cir.bool, !s32i, !s32i) -> !cir.vector<4 x !cir.f16>
// LLVM: define{{.*}} <4 x half> @_Z27test_sample_l_1darray_v4f16fffu18__amdgpu_texture_tDv4_i(
// LLVM: call {{.*}}<4 x half> @llvm.amdgcn.image.sample.l.1darray.v4f16.f32.v8i32.v4i32(i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ half4 test_sample_l_1darray_v4f16(float s, float slice, float lod, __amdgpu_texture_t tex, int4 samp) {
  return __builtin_amdgcn_image_sample_l_1darray_v4f16_f32(15, s, slice, lod, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @_Z27test_sample_d_1darray_v4f16
// CIR: cir.call_llvm_intrinsic "amdgcn.image.sample.d.1darray" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.float, !cir.float, !cir.vector<8 x !s32i>, !cir.vector<4 x !s32i>, !cir.bool, !s32i, !s32i) -> !cir.vector<4 x !cir.f16>
// LLVM: define{{.*}} <4 x half> @_Z27test_sample_d_1darray_v4f16ffffu18__amdgpu_texture_tDv4_i(
// LLVM: call {{.*}}<4 x half> @llvm.amdgcn.image.sample.d.1darray.v4f16.f32.f32.v8i32.v4i32(i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ half4 test_sample_d_1darray_v4f16(float dsdx, float dsdy, float s, float slice, __amdgpu_texture_t tex, int4 samp) {
  return __builtin_amdgcn_image_sample_d_1darray_v4f16_f32(15, dsdx, dsdy, s, slice, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @_Z28test_sample_lz_2darray_v4f16
// CIR: cir.call_llvm_intrinsic "amdgcn.image.sample.lz.2darray" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.float, !cir.vector<8 x !s32i>, !cir.vector<4 x !s32i>, !cir.bool, !s32i, !s32i) -> !cir.vector<4 x !cir.f16>
// LLVM: define{{.*}} <4 x half> @_Z28test_sample_lz_2darray_v4f16fffu18__amdgpu_texture_tDv4_i(
// LLVM: call {{.*}}<4 x half> @llvm.amdgcn.image.sample.lz.2darray.v4f16.f32.v8i32.v4i32(i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ half4 test_sample_lz_2darray_v4f16(float s, float t, float slice, __amdgpu_texture_t tex, int4 samp) {
  return __builtin_amdgcn_image_sample_lz_2darray_v4f16_f32(15, s, t, slice, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @_Z27test_sample_l_2darray_v4f16
// CIR: cir.call_llvm_intrinsic "amdgcn.image.sample.l.2darray" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.float, !cir.float, !cir.vector<8 x !s32i>, !cir.vector<4 x !s32i>, !cir.bool, !s32i, !s32i) -> !cir.vector<4 x !cir.f16>
// LLVM: define{{.*}} <4 x half> @_Z27test_sample_l_2darray_v4f16ffffu18__amdgpu_texture_tDv4_i(
// LLVM: call {{.*}}<4 x half> @llvm.amdgcn.image.sample.l.2darray.v4f16.f32.v8i32.v4i32(i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ half4 test_sample_l_2darray_v4f16(float s, float t, float slice, float lod, __amdgpu_texture_t tex, int4 samp) {
  return __builtin_amdgcn_image_sample_l_2darray_v4f16_f32(15, s, t, slice, lod, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @_Z27test_sample_d_2darray_v4f16
// CIR: cir.call_llvm_intrinsic "amdgcn.image.sample.d.2darray" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.float, !cir.float, !cir.float, !cir.float, !cir.float, !cir.vector<8 x !s32i>, !cir.vector<4 x !s32i>, !cir.bool, !s32i, !s32i) -> !cir.vector<4 x !cir.f16>
// LLVM: define{{.*}} <4 x half> @_Z27test_sample_d_2darray_v4f16fffffffu18__amdgpu_texture_tDv4_i(
// LLVM: call {{.*}}<4 x half> @llvm.amdgcn.image.sample.d.2darray.v4f16.f32.f32.v8i32.v4i32(i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ half4 test_sample_d_2darray_v4f16(float dsdx, float dtdx, float dsdy, float dtdy, float s, float t, float slice, __amdgpu_texture_t tex, int4 samp) {
  return __builtin_amdgcn_image_sample_d_2darray_v4f16_f32(15, dsdx, dtdx, dsdy, dtdy, s, t, slice, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @_Z21test_sample_lz_2d_f32
// CIR: cir.call_llvm_intrinsic "amdgcn.image.sample.lz.2d" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.vector<8 x !s32i>, !cir.vector<4 x !s32i>, !cir.bool, !s32i, !s32i) -> !cir.float
// LLVM: define{{.*}} float @_Z21test_sample_lz_2d_f32ffu18__amdgpu_texture_tDv4_i(
// LLVM: call {{.*}}float @llvm.amdgcn.image.sample.lz.2d.f32.f32.v8i32.v4i32(i32 {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ float test_sample_lz_2d_f32(float s, float t, __amdgpu_texture_t tex, int4 samp) {
  return __builtin_amdgcn_image_sample_lz_2d_f32_f32(1, s, t, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @_Z20test_sample_l_2d_f32
// CIR: cir.call_llvm_intrinsic "amdgcn.image.sample.l.2d" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.float, !cir.vector<8 x !s32i>, !cir.vector<4 x !s32i>, !cir.bool, !s32i, !s32i) -> !cir.float
// LLVM: define{{.*}} float @_Z20test_sample_l_2d_f32fffu18__amdgpu_texture_tDv4_i(
// LLVM: call {{.*}}float @llvm.amdgcn.image.sample.l.2d.f32.f32.v8i32.v4i32(i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ float test_sample_l_2d_f32(float s, float t, float lod, __amdgpu_texture_t tex, int4 samp) {
  return __builtin_amdgcn_image_sample_l_2d_f32_f32(1, s, t, lod, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @_Z20test_sample_d_2d_f32
// CIR: cir.call_llvm_intrinsic "amdgcn.image.sample.d.2d" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.float, !cir.float, !cir.float, !cir.float, !cir.vector<8 x !s32i>, !cir.vector<4 x !s32i>, !cir.bool, !s32i, !s32i) -> !cir.float
// LLVM: define{{.*}} float @_Z20test_sample_d_2d_f32ffffffu18__amdgpu_texture_tDv4_i(
// LLVM: call {{.*}}float @llvm.amdgcn.image.sample.d.2d.f32.f32.f32.v8i32.v4i32(i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ float test_sample_d_2d_f32(float dsdx, float dtdx, float dsdy, float dtdy, float s, float t, __amdgpu_texture_t tex, int4 samp) {
  return __builtin_amdgcn_image_sample_d_2d_f32_f32(1, dsdx, dtdx, dsdy, dtdy, s, t, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @_Z26test_sample_lz_2darray_f32
// CIR: cir.call_llvm_intrinsic "amdgcn.image.sample.lz.2darray" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.float, !cir.vector<8 x !s32i>, !cir.vector<4 x !s32i>, !cir.bool, !s32i, !s32i) -> !cir.float
// LLVM: define{{.*}} float @_Z26test_sample_lz_2darray_f32fffu18__amdgpu_texture_tDv4_i(
// LLVM: call {{.*}}float @llvm.amdgcn.image.sample.lz.2darray.f32.f32.v8i32.v4i32(i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ float test_sample_lz_2darray_f32(float s, float t, float slice, __amdgpu_texture_t tex, int4 samp) {
  return __builtin_amdgcn_image_sample_lz_2darray_f32_f32(1, s, t, slice, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @_Z25test_sample_l_2darray_f32
// CIR: cir.call_llvm_intrinsic "amdgcn.image.sample.l.2darray" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.float, !cir.float, !cir.vector<8 x !s32i>, !cir.vector<4 x !s32i>, !cir.bool, !s32i, !s32i) -> !cir.float
// LLVM: define{{.*}} float @_Z25test_sample_l_2darray_f32ffffu18__amdgpu_texture_tDv4_i(
// LLVM: call {{.*}}float @llvm.amdgcn.image.sample.l.2darray.f32.f32.v8i32.v4i32(i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ float test_sample_l_2darray_f32(float s, float t, float slice, float lod, __amdgpu_texture_t tex, int4 samp) {
  return __builtin_amdgcn_image_sample_l_2darray_f32_f32(1, s, t, slice, lod, tex, samp, 0, 120, 110);
}

// CIR-LABEL: @_Z25test_sample_d_2darray_f32
// CIR: cir.call_llvm_intrinsic "amdgcn.image.sample.d.2darray" {{.*}} : (!s32i, !cir.float, !cir.float, !cir.float, !cir.float, !cir.float, !cir.float, !cir.float, !cir.vector<8 x !s32i>, !cir.vector<4 x !s32i>, !cir.bool, !s32i, !s32i) -> !cir.float
// LLVM: define{{.*}} float @_Z25test_sample_d_2darray_f32fffffffu18__amdgpu_texture_tDv4_i(
// LLVM: call {{.*}}float @llvm.amdgcn.image.sample.d.2darray.f32.f32.f32.v8i32.v4i32(i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, <8 x i32> {{.*}}, <4 x i32> {{.*}}, i1 {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ float test_sample_d_2darray_f32(float dsdx, float dtdx, float dsdy, float dtdy, float s, float t, float slice, __amdgpu_texture_t tex, int4 samp) {
  return __builtin_amdgcn_image_sample_d_2darray_f32_f32(1, dsdx, dtdx, dsdy, dtdy, s, t, slice, tex, samp, 0, 120, 110);
}
