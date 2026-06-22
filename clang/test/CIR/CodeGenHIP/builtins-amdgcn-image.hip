#include "../CodeGenCUDA/Inputs/cuda.h"

// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -target-cpu gfx1100 -fcuda-is-device -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -target-cpu gfx1100 -fcuda-is-device -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 \
// RUN:            -target-cpu gfx1100 -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

typedef float float4 __attribute__((ext_vector_type(4)));
typedef _Float16 half;
typedef half half4 __attribute__((ext_vector_type(4)));

// CIR-LABEL: @_Z24test_image_load_1d_v4f32
// CIR: cir.call_llvm_intrinsic "amdgcn.image.load.1d" {{.*}} : (!s32i, !s32i, !cir.vector<8 x !s32i>, !s32i, !s32i) -> !cir.vector<4 x !cir.float>
// LLVM: define{{.*}} <4 x float> @_Z24test_image_load_1d_v4f32iu18__amdgpu_texture_t(
// LLVM: call {{.*}}<4 x float> @llvm.amdgcn.image.load.1d.v4f32.i32.v8i32(i32 {{.*}}, i32 {{.*}}, <8 x i32> {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ float4 test_image_load_1d_v4f32(int x, __amdgpu_texture_t rsrc) {
  return __builtin_amdgcn_image_load_1d_v4f32_i32(15, x, rsrc, 120, 110);
}

// CIR-LABEL: @_Z24test_image_load_1d_v4f16
// CIR: cir.call_llvm_intrinsic "amdgcn.image.load.1d" {{.*}} : (!s32i, !s32i, !cir.vector<8 x !s32i>, !s32i, !s32i) -> !cir.vector<4 x !cir.f16>
// LLVM: define{{.*}} <4 x half> @_Z24test_image_load_1d_v4f16iu18__amdgpu_texture_t(
// LLVM: call {{.*}}<4 x half> @llvm.amdgcn.image.load.1d.v4f16.i32.v8i32(i32 {{.*}}, i32 {{.*}}, <8 x i32> {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ half4 test_image_load_1d_v4f16(int x, __amdgpu_texture_t rsrc) {
  return __builtin_amdgcn_image_load_1d_v4f16_i32(15, x, rsrc, 120, 110);
}

// CIR-LABEL: @_Z25test_image_store_1d_v4f32
// CIR: cir.call_llvm_intrinsic "amdgcn.image.store.1d" {{.*}} : (!cir.vector<4 x !cir.float>, !s32i, !s32i, !cir.vector<8 x !s32i>, !s32i, !s32i) -> !void
// LLVM: define{{.*}} void @_Z25test_image_store_1d_v4f32Dv4_fiu18__amdgpu_texture_t(
// LLVM: call void @llvm.amdgcn.image.store.1d.v4f32.i32.v8i32(<4 x float> {{.*}}, i32 {{.*}}, i32 {{.*}}, <8 x i32> {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ void test_image_store_1d_v4f32(float4 val, int x, __amdgpu_texture_t rsrc) {
  __builtin_amdgcn_image_store_1d_v4f32_i32(val, 15, x, rsrc, 120, 110);
}

// CIR-LABEL: @_Z25test_image_store_1d_v4f16
// CIR: cir.call_llvm_intrinsic "amdgcn.image.store.1d" {{.*}} : (!cir.vector<4 x !cir.f16>, !s32i, !s32i, !cir.vector<8 x !s32i>, !s32i, !s32i) -> !void
// LLVM: define{{.*}} void @_Z25test_image_store_1d_v4f16Dv4_DF16_iu18__amdgpu_texture_t(
// LLVM: call void @llvm.amdgcn.image.store.1d.v4f16.i32.v8i32(<4 x half> {{.*}}, i32 {{.*}}, i32 {{.*}}, <8 x i32> {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ void test_image_store_1d_v4f16(half4 val, int x, __amdgpu_texture_t rsrc) {
  __builtin_amdgcn_image_store_1d_v4f16_i32(val, 15, x, rsrc, 120, 110);
}

// CIR-LABEL: @_Z29test_image_load_1darray_v4f32
// CIR: cir.call_llvm_intrinsic "amdgcn.image.load.1darray" {{.*}} : (!s32i, !s32i, !s32i, !cir.vector<8 x !s32i>, !s32i, !s32i) -> !cir.vector<4 x !cir.float>
// LLVM: define{{.*}} <4 x float> @_Z29test_image_load_1darray_v4f32iiu18__amdgpu_texture_t(
// LLVM: call {{.*}}<4 x float> @llvm.amdgcn.image.load.1darray.v4f32.i32.v8i32(i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, <8 x i32> {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ float4 test_image_load_1darray_v4f32(int x, int slice, __amdgpu_texture_t rsrc) {
  return __builtin_amdgcn_image_load_1darray_v4f32_i32(15, x, slice, rsrc, 120, 110);
}

// CIR-LABEL: @_Z29test_image_load_1darray_v4f16
// CIR: cir.call_llvm_intrinsic "amdgcn.image.load.1darray" {{.*}} : (!s32i, !s32i, !s32i, !cir.vector<8 x !s32i>, !s32i, !s32i) -> !cir.vector<4 x !cir.f16>
// LLVM: define{{.*}} <4 x half> @_Z29test_image_load_1darray_v4f16iiu18__amdgpu_texture_t(
// LLVM: call {{.*}}<4 x half> @llvm.amdgcn.image.load.1darray.v4f16.i32.v8i32(i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, <8 x i32> {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ half4 test_image_load_1darray_v4f16(int x, int slice, __amdgpu_texture_t rsrc) {
  return __builtin_amdgcn_image_load_1darray_v4f16_i32(15, x, slice, rsrc, 120, 110);
}

// CIR-LABEL: @_Z30test_image_store_1darray_v4f32
// CIR: cir.call_llvm_intrinsic "amdgcn.image.store.1darray" {{.*}} : (!cir.vector<4 x !cir.float>, !s32i, !s32i, !s32i, !cir.vector<8 x !s32i>, !s32i, !s32i) -> !void
// LLVM: define{{.*}} void @_Z30test_image_store_1darray_v4f32Dv4_fiiu18__amdgpu_texture_t(
// LLVM: call void @llvm.amdgcn.image.store.1darray.v4f32.i32.v8i32(<4 x float> {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, <8 x i32> {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ void test_image_store_1darray_v4f32(float4 val, int x, int slice, __amdgpu_texture_t rsrc) {
  __builtin_amdgcn_image_store_1darray_v4f32_i32(val, 15, x, slice, rsrc, 120, 110);
}

// CIR-LABEL: @_Z30test_image_store_1darray_v4f16
// CIR: cir.call_llvm_intrinsic "amdgcn.image.store.1darray" {{.*}} : (!cir.vector<4 x !cir.f16>, !s32i, !s32i, !s32i, !cir.vector<8 x !s32i>, !s32i, !s32i) -> !void
// LLVM: define{{.*}} void @_Z30test_image_store_1darray_v4f16Dv4_DF16_iiu18__amdgpu_texture_t(
// LLVM: call void @llvm.amdgcn.image.store.1darray.v4f16.i32.v8i32(<4 x half> {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, <8 x i32> {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ void test_image_store_1darray_v4f16(half4 val, int x, int slice, __amdgpu_texture_t rsrc) {
  __builtin_amdgcn_image_store_1darray_v4f16_i32(val, 15, x, slice, rsrc, 120, 110);
}

// CIR-LABEL: @_Z22test_image_load_2d_f32
// CIR: cir.call_llvm_intrinsic "amdgcn.image.load.2d" {{.*}} : (!s32i, !s32i, !s32i, !cir.vector<8 x !s32i>, !s32i, !s32i) -> !cir.float
// LLVM: define{{.*}} float @_Z22test_image_load_2d_f32iiu18__amdgpu_texture_t(
// LLVM: call {{.*}}float @llvm.amdgcn.image.load.2d.f32.i32.v8i32(i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, <8 x i32> {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ float test_image_load_2d_f32(int x, int y, __amdgpu_texture_t rsrc) {
  return __builtin_amdgcn_image_load_2d_f32_i32(2, x, y, rsrc, 106, 103);
}

// CIR-LABEL: @_Z24test_image_load_2d_v4f32
// CIR: cir.call_llvm_intrinsic "amdgcn.image.load.2d" {{.*}} : (!s32i, !s32i, !s32i, !cir.vector<8 x !s32i>, !s32i, !s32i) -> !cir.vector<4 x !cir.float>
// LLVM: define{{.*}} <4 x float> @_Z24test_image_load_2d_v4f32iiu18__amdgpu_texture_t(
// LLVM: call {{.*}}<4 x float> @llvm.amdgcn.image.load.2d.v4f32.i32.v8i32(i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, <8 x i32> {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ float4 test_image_load_2d_v4f32(int x, int y, __amdgpu_texture_t rsrc) {
  return __builtin_amdgcn_image_load_2d_v4f32_i32(15, x, y, rsrc, 120, 110);
}

// CIR-LABEL: @_Z24test_image_load_2d_v4f16
// CIR: cir.call_llvm_intrinsic "amdgcn.image.load.2d" {{.*}} : (!s32i, !s32i, !s32i, !cir.vector<8 x !s32i>, !s32i, !s32i) -> !cir.vector<4 x !cir.f16>
// LLVM: define{{.*}} <4 x half> @_Z24test_image_load_2d_v4f16iiu18__amdgpu_texture_t(
// LLVM: call {{.*}}<4 x half> @llvm.amdgcn.image.load.2d.v4f16.i32.v8i32(i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, <8 x i32> {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ half4 test_image_load_2d_v4f16(int x, int y, __amdgpu_texture_t rsrc) {
  return __builtin_amdgcn_image_load_2d_v4f16_i32(15, x, y, rsrc, 120, 110);
}

// CIR-LABEL: @_Z23test_image_store_2d_f32
// CIR: cir.call_llvm_intrinsic "amdgcn.image.store.2d" {{.*}} : (!cir.float, !s32i, !s32i, !s32i, !cir.vector<8 x !s32i>, !s32i, !s32i) -> !void
// LLVM: define{{.*}} void @_Z23test_image_store_2d_f32fiiu18__amdgpu_texture_t(
// LLVM: call void @llvm.amdgcn.image.store.2d.f32.i32.v8i32(float {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, <8 x i32> {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ void test_image_store_2d_f32(float val, int x, int y, __amdgpu_texture_t rsrc) {
  __builtin_amdgcn_image_store_2d_f32_i32(val, 12, x, y, rsrc, 106, 103);
}

// CIR-LABEL: @_Z25test_image_store_2d_v4f32
// CIR: cir.call_llvm_intrinsic "amdgcn.image.store.2d" {{.*}} : (!cir.vector<4 x !cir.float>, !s32i, !s32i, !s32i, !cir.vector<8 x !s32i>, !s32i, !s32i) -> !void
// LLVM: define{{.*}} void @_Z25test_image_store_2d_v4f32Dv4_fiiu18__amdgpu_texture_t(
// LLVM: call void @llvm.amdgcn.image.store.2d.v4f32.i32.v8i32(<4 x float> {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, <8 x i32> {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ void test_image_store_2d_v4f32(float4 val, int x, int y, __amdgpu_texture_t rsrc) {
  __builtin_amdgcn_image_store_2d_v4f32_i32(val, 15, x, y, rsrc, 120, 110);
}

// CIR-LABEL: @_Z25test_image_store_2d_v4f16
// CIR: cir.call_llvm_intrinsic "amdgcn.image.store.2d" {{.*}} : (!cir.vector<4 x !cir.f16>, !s32i, !s32i, !s32i, !cir.vector<8 x !s32i>, !s32i, !s32i) -> !void
// LLVM: define{{.*}} void @_Z25test_image_store_2d_v4f16Dv4_DF16_iiu18__amdgpu_texture_t(
// LLVM: call void @llvm.amdgcn.image.store.2d.v4f16.i32.v8i32(<4 x half> {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, <8 x i32> {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ void test_image_store_2d_v4f16(half4 val, int x, int y, __amdgpu_texture_t rsrc) {
  __builtin_amdgcn_image_store_2d_v4f16_i32(val, 15, x, y, rsrc, 120, 110);
}

// CIR-LABEL: @_Z27test_image_load_2darray_f32
// CIR: cir.call_llvm_intrinsic "amdgcn.image.load.2darray" {{.*}} : (!s32i, !s32i, !s32i, !s32i, !cir.vector<8 x !s32i>, !s32i, !s32i) -> !cir.float
// LLVM: define{{.*}} float @_Z27test_image_load_2darray_f32iiiu18__amdgpu_texture_t(
// LLVM: call {{.*}}float @llvm.amdgcn.image.load.2darray.f32.i32.v8i32(i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, <8 x i32> {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ float test_image_load_2darray_f32(int x, int y, int slice, __amdgpu_texture_t rsrc) {
  return __builtin_amdgcn_image_load_2darray_f32_i32(4, x, y, slice, rsrc, 120, 110);
}

// CIR-LABEL: @_Z29test_image_load_2darray_v4f32
// CIR: cir.call_llvm_intrinsic "amdgcn.image.load.2darray" {{.*}} : (!s32i, !s32i, !s32i, !s32i, !cir.vector<8 x !s32i>, !s32i, !s32i) -> !cir.vector<4 x !cir.float>
// LLVM: define{{.*}} <4 x float> @_Z29test_image_load_2darray_v4f32iiiu18__amdgpu_texture_t(
// LLVM: call {{.*}}<4 x float> @llvm.amdgcn.image.load.2darray.v4f32.i32.v8i32(i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, <8 x i32> {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ float4 test_image_load_2darray_v4f32(int x, int y, int slice, __amdgpu_texture_t rsrc) {
  return __builtin_amdgcn_image_load_2darray_v4f32_i32(15, x, y, slice, rsrc, 120, 110);
}

// CIR-LABEL: @_Z29test_image_load_2darray_v4f16
// CIR: cir.call_llvm_intrinsic "amdgcn.image.load.2darray" {{.*}} : (!s32i, !s32i, !s32i, !s32i, !cir.vector<8 x !s32i>, !s32i, !s32i) -> !cir.vector<4 x !cir.f16>
// LLVM: define{{.*}} <4 x half> @_Z29test_image_load_2darray_v4f16iiiu18__amdgpu_texture_t(
// LLVM: call {{.*}}<4 x half> @llvm.amdgcn.image.load.2darray.v4f16.i32.v8i32(i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, <8 x i32> {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ half4 test_image_load_2darray_v4f16(int x, int y, int slice, __amdgpu_texture_t rsrc) {
  return __builtin_amdgcn_image_load_2darray_v4f16_i32(15, x, y, slice, rsrc, 120, 110);
}

// CIR-LABEL: @_Z28test_image_store_2darray_f32
// CIR: cir.call_llvm_intrinsic "amdgcn.image.store.2darray" {{.*}} : (!cir.float, !s32i, !s32i, !s32i, !s32i, !cir.vector<8 x !s32i>, !s32i, !s32i) -> !void
// LLVM: define{{.*}} void @_Z28test_image_store_2darray_f32fiiiu18__amdgpu_texture_t(
// LLVM: call void @llvm.amdgcn.image.store.2darray.f32.i32.v8i32(float {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, <8 x i32> {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ void test_image_store_2darray_f32(float val, int x, int y, int slice, __amdgpu_texture_t rsrc) {
  __builtin_amdgcn_image_store_2darray_f32_i32(val, 12, x, y, slice, rsrc, 106, 103);
}

// CIR-LABEL: @_Z30test_image_store_2darray_v4f32
// CIR: cir.call_llvm_intrinsic "amdgcn.image.store.2darray" {{.*}} : (!cir.vector<4 x !cir.float>, !s32i, !s32i, !s32i, !s32i, !cir.vector<8 x !s32i>, !s32i, !s32i) -> !void
// LLVM: define{{.*}} void @_Z30test_image_store_2darray_v4f32Dv4_fiiiu18__amdgpu_texture_t(
// LLVM: call void @llvm.amdgcn.image.store.2darray.v4f32.i32.v8i32(<4 x float> {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, <8 x i32> {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ void test_image_store_2darray_v4f32(float4 val, int x, int y, int slice, __amdgpu_texture_t rsrc) {
  __builtin_amdgcn_image_store_2darray_v4f32_i32(val, 15, x, y, slice, rsrc, 120, 110);
}

// CIR-LABEL: @_Z30test_image_store_2darray_v4f16
// CIR: cir.call_llvm_intrinsic "amdgcn.image.store.2darray" {{.*}} : (!cir.vector<4 x !cir.f16>, !s32i, !s32i, !s32i, !s32i, !cir.vector<8 x !s32i>, !s32i, !s32i) -> !void
// LLVM: define{{.*}} void @_Z30test_image_store_2darray_v4f16Dv4_DF16_iiiu18__amdgpu_texture_t(
// LLVM: call void @llvm.amdgcn.image.store.2darray.v4f16.i32.v8i32(<4 x half> {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, <8 x i32> {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ void test_image_store_2darray_v4f16(half4 val, int x, int y, int slice, __amdgpu_texture_t rsrc) {
  __builtin_amdgcn_image_store_2darray_v4f16_i32(val, 15, x, y, slice, rsrc, 120, 110);
}

// CIR-LABEL: @_Z24test_image_load_3d_v4f32
// CIR: cir.call_llvm_intrinsic "amdgcn.image.load.3d" {{.*}} : (!s32i, !s32i, !s32i, !s32i, !cir.vector<8 x !s32i>, !s32i, !s32i) -> !cir.vector<4 x !cir.float>
// LLVM: define{{.*}} <4 x float> @_Z24test_image_load_3d_v4f32iiiu18__amdgpu_texture_t(
// LLVM: call {{.*}}<4 x float> @llvm.amdgcn.image.load.3d.v4f32.i32.v8i32(i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, <8 x i32> {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ float4 test_image_load_3d_v4f32(int x, int y, int z, __amdgpu_texture_t rsrc) {
  return __builtin_amdgcn_image_load_3d_v4f32_i32(15, x, y, z, rsrc, 120, 110);
}

// CIR-LABEL: @_Z24test_image_load_3d_v4f16
// CIR: cir.call_llvm_intrinsic "amdgcn.image.load.3d" {{.*}} : (!s32i, !s32i, !s32i, !s32i, !cir.vector<8 x !s32i>, !s32i, !s32i) -> !cir.vector<4 x !cir.f16>
// LLVM: define{{.*}} <4 x half> @_Z24test_image_load_3d_v4f16iiiu18__amdgpu_texture_t(
// LLVM: call {{.*}}<4 x half> @llvm.amdgcn.image.load.3d.v4f16.i32.v8i32(i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, <8 x i32> {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ half4 test_image_load_3d_v4f16(int x, int y, int z, __amdgpu_texture_t rsrc) {
  return __builtin_amdgcn_image_load_3d_v4f16_i32(15, x, y, z, rsrc, 120, 110);
}

// CIR-LABEL: @_Z25test_image_store_3d_v4f32
// CIR: cir.call_llvm_intrinsic "amdgcn.image.store.3d" {{.*}} : (!cir.vector<4 x !cir.float>, !s32i, !s32i, !s32i, !s32i, !cir.vector<8 x !s32i>, !s32i, !s32i) -> !void
// LLVM: define{{.*}} void @_Z25test_image_store_3d_v4f32Dv4_fiiiu18__amdgpu_texture_t(
// LLVM: call void @llvm.amdgcn.image.store.3d.v4f32.i32.v8i32(<4 x float> {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, <8 x i32> {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ void test_image_store_3d_v4f32(float4 val, int x, int y, int z, __amdgpu_texture_t rsrc) {
  __builtin_amdgcn_image_store_3d_v4f32_i32(val, 15, x, y, z, rsrc, 120, 110);
}

// CIR-LABEL: @_Z25test_image_store_3d_v4f16
// CIR: cir.call_llvm_intrinsic "amdgcn.image.store.3d" {{.*}} : (!cir.vector<4 x !cir.f16>, !s32i, !s32i, !s32i, !s32i, !cir.vector<8 x !s32i>, !s32i, !s32i) -> !void
// LLVM: define{{.*}} void @_Z25test_image_store_3d_v4f16Dv4_DF16_iiiu18__amdgpu_texture_t(
// LLVM: call void @llvm.amdgcn.image.store.3d.v4f16.i32.v8i32(<4 x half> {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, <8 x i32> {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ void test_image_store_3d_v4f16(half4 val, int x, int y, int z, __amdgpu_texture_t rsrc) {
  __builtin_amdgcn_image_store_3d_v4f16_i32(val, 15, x, y, z, rsrc, 120, 110);
}

// CIR-LABEL: @_Z26test_image_load_cube_v4f32
// CIR: cir.call_llvm_intrinsic "amdgcn.image.load.cube" {{.*}} : (!s32i, !s32i, !s32i, !s32i, !cir.vector<8 x !s32i>, !s32i, !s32i) -> !cir.vector<4 x !cir.float>
// LLVM: define{{.*}} <4 x float> @_Z26test_image_load_cube_v4f32iiiu18__amdgpu_texture_t(
// LLVM: call {{.*}}<4 x float> @llvm.amdgcn.image.load.cube.v4f32.i32.v8i32(i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, <8 x i32> {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ float4 test_image_load_cube_v4f32(int x, int y, int face, __amdgpu_texture_t rsrc) {
  return __builtin_amdgcn_image_load_cube_v4f32_i32(15, x, y, face, rsrc, 120, 110);
}

// CIR-LABEL: @_Z26test_image_load_cube_v4f16
// CIR: cir.call_llvm_intrinsic "amdgcn.image.load.cube" {{.*}} : (!s32i, !s32i, !s32i, !s32i, !cir.vector<8 x !s32i>, !s32i, !s32i) -> !cir.vector<4 x !cir.f16>
// LLVM: define{{.*}} <4 x half> @_Z26test_image_load_cube_v4f16iiiu18__amdgpu_texture_t(
// LLVM: call {{.*}}<4 x half> @llvm.amdgcn.image.load.cube.v4f16.i32.v8i32(i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, <8 x i32> {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ half4 test_image_load_cube_v4f16(int x, int y, int face, __amdgpu_texture_t rsrc) {
  return __builtin_amdgcn_image_load_cube_v4f16_i32(15, x, y, face, rsrc, 120, 110);
}

// CIR-LABEL: @_Z27test_image_store_cube_v4f32
// CIR: cir.call_llvm_intrinsic "amdgcn.image.store.cube" {{.*}} : (!cir.vector<4 x !cir.float>, !s32i, !s32i, !s32i, !s32i, !cir.vector<8 x !s32i>, !s32i, !s32i) -> !void
// LLVM: define{{.*}} void @_Z27test_image_store_cube_v4f32Dv4_fiiiu18__amdgpu_texture_t(
// LLVM: call void @llvm.amdgcn.image.store.cube.v4f32.i32.v8i32(<4 x float> {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, <8 x i32> {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ void test_image_store_cube_v4f32(float4 val, int x, int y, int face, __amdgpu_texture_t rsrc) {
  __builtin_amdgcn_image_store_cube_v4f32_i32(val, 15, x, y, face, rsrc, 120, 110);
}

// CIR-LABEL: @_Z27test_image_store_cube_v4f16
// CIR: cir.call_llvm_intrinsic "amdgcn.image.store.cube" {{.*}} : (!cir.vector<4 x !cir.f16>, !s32i, !s32i, !s32i, !s32i, !cir.vector<8 x !s32i>, !s32i, !s32i) -> !void
// LLVM: define{{.*}} void @_Z27test_image_store_cube_v4f16Dv4_DF16_iiiu18__amdgpu_texture_t(
// LLVM: call void @llvm.amdgcn.image.store.cube.v4f16.i32.v8i32(<4 x half> {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, <8 x i32> {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ void test_image_store_cube_v4f16(half4 val, int x, int y, int face, __amdgpu_texture_t rsrc) {
  __builtin_amdgcn_image_store_cube_v4f16_i32(val, 15, x, y, face, rsrc, 120, 110);
}

// CIR-LABEL: @_Z28test_image_load_mip_1d_v4f32
// CIR: cir.call_llvm_intrinsic "amdgcn.image.load.mip.1d" {{.*}} : (!s32i, !s32i, !s32i, !cir.vector<8 x !s32i>, !s32i, !s32i) -> !cir.vector<4 x !cir.float>
// LLVM: define{{.*}} <4 x float> @_Z28test_image_load_mip_1d_v4f32iiu18__amdgpu_texture_t(
// LLVM: call {{.*}}<4 x float> @llvm.amdgcn.image.load.mip.1d.v4f32.i32.v8i32(i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, <8 x i32> {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ float4 test_image_load_mip_1d_v4f32(int x, int mip, __amdgpu_texture_t rsrc) {
  return __builtin_amdgcn_image_load_mip_1d_v4f32_i32(15, x, mip, rsrc, 120, 110);
}

// CIR-LABEL: @_Z28test_image_load_mip_1d_v4f16
// CIR: cir.call_llvm_intrinsic "amdgcn.image.load.mip.1d" {{.*}} : (!s32i, !s32i, !s32i, !cir.vector<8 x !s32i>, !s32i, !s32i) -> !cir.vector<4 x !cir.f16>
// LLVM: define{{.*}} <4 x half> @_Z28test_image_load_mip_1d_v4f16iiu18__amdgpu_texture_t(
// LLVM: call {{.*}}<4 x half> @llvm.amdgcn.image.load.mip.1d.v4f16.i32.v8i32(i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, <8 x i32> {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ half4 test_image_load_mip_1d_v4f16(int x, int mip, __amdgpu_texture_t rsrc) {
  return __builtin_amdgcn_image_load_mip_1d_v4f16_i32(15, x, mip, rsrc, 120, 110);
}

// CIR-LABEL: @_Z29test_image_store_mip_1d_v4f32
// CIR: cir.call_llvm_intrinsic "amdgcn.image.store.mip.1d" {{.*}} : (!cir.vector<4 x !cir.float>, !s32i, !s32i, !s32i, !cir.vector<8 x !s32i>, !s32i, !s32i) -> !void
// LLVM: define{{.*}} void @_Z29test_image_store_mip_1d_v4f32Dv4_fiiu18__amdgpu_texture_t(
// LLVM: call void @llvm.amdgcn.image.store.mip.1d.v4f32.i32.v8i32(<4 x float> {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, <8 x i32> {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ void test_image_store_mip_1d_v4f32(float4 val, int x, int mip, __amdgpu_texture_t rsrc) {
  __builtin_amdgcn_image_store_mip_1d_v4f32_i32(val, 15, x, mip, rsrc, 120, 110);
}

// CIR-LABEL: @_Z29test_image_store_mip_1d_v4f16
// CIR: cir.call_llvm_intrinsic "amdgcn.image.store.mip.1d" {{.*}} : (!cir.vector<4 x !cir.f16>, !s32i, !s32i, !s32i, !cir.vector<8 x !s32i>, !s32i, !s32i) -> !void
// LLVM: define{{.*}} void @_Z29test_image_store_mip_1d_v4f16Dv4_DF16_iiu18__amdgpu_texture_t(
// LLVM: call void @llvm.amdgcn.image.store.mip.1d.v4f16.i32.v8i32(<4 x half> {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, <8 x i32> {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ void test_image_store_mip_1d_v4f16(half4 val, int x, int mip, __amdgpu_texture_t rsrc) {
  __builtin_amdgcn_image_store_mip_1d_v4f16_i32(val, 15, x, mip, rsrc, 120, 110);
}

// CIR-LABEL: @_Z33test_image_load_mip_1darray_v4f32
// CIR: cir.call_llvm_intrinsic "amdgcn.image.load.mip.1darray" {{.*}} : (!s32i, !s32i, !s32i, !s32i, !cir.vector<8 x !s32i>, !s32i, !s32i) -> !cir.vector<4 x !cir.float>
// LLVM: define{{.*}} <4 x float> @_Z33test_image_load_mip_1darray_v4f32iiiu18__amdgpu_texture_t(
// LLVM: call {{.*}}<4 x float> @llvm.amdgcn.image.load.mip.1darray.v4f32.i32.v8i32(i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, <8 x i32> {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ float4 test_image_load_mip_1darray_v4f32(int x, int slice, int mip, __amdgpu_texture_t rsrc) {
  return __builtin_amdgcn_image_load_mip_1darray_v4f32_i32(15, x, slice, mip, rsrc, 120, 110);
}

// CIR-LABEL: @_Z33test_image_load_mip_1darray_v4f16
// CIR: cir.call_llvm_intrinsic "amdgcn.image.load.mip.1darray" {{.*}} : (!s32i, !s32i, !s32i, !s32i, !cir.vector<8 x !s32i>, !s32i, !s32i) -> !cir.vector<4 x !cir.f16>
// LLVM: define{{.*}} <4 x half> @_Z33test_image_load_mip_1darray_v4f16iiiu18__amdgpu_texture_t(
// LLVM: call {{.*}}<4 x half> @llvm.amdgcn.image.load.mip.1darray.v4f16.i32.v8i32(i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, <8 x i32> {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ half4 test_image_load_mip_1darray_v4f16(int x, int slice, int mip, __amdgpu_texture_t rsrc) {
  return __builtin_amdgcn_image_load_mip_1darray_v4f16_i32(15, x, slice, mip, rsrc, 120, 110);
}

// CIR-LABEL: @_Z34test_image_store_mip_1darray_v4f32
// CIR: cir.call_llvm_intrinsic "amdgcn.image.store.mip.1darray" {{.*}} : (!cir.vector<4 x !cir.float>, !s32i, !s32i, !s32i, !s32i, !cir.vector<8 x !s32i>, !s32i, !s32i) -> !void
// LLVM: define{{.*}} void @_Z34test_image_store_mip_1darray_v4f32Dv4_fiiiu18__amdgpu_texture_t(
// LLVM: call void @llvm.amdgcn.image.store.mip.1darray.v4f32.i32.v8i32(<4 x float> {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, <8 x i32> {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ void test_image_store_mip_1darray_v4f32(float4 val, int x, int slice, int mip, __amdgpu_texture_t rsrc) {
  __builtin_amdgcn_image_store_mip_1darray_v4f32_i32(val, 15, x, slice, mip, rsrc, 120, 110);
}

// CIR-LABEL: @_Z34test_image_store_mip_1darray_v4f16
// CIR: cir.call_llvm_intrinsic "amdgcn.image.store.mip.1darray" {{.*}} : (!cir.vector<4 x !cir.f16>, !s32i, !s32i, !s32i, !s32i, !cir.vector<8 x !s32i>, !s32i, !s32i) -> !void
// LLVM: define{{.*}} void @_Z34test_image_store_mip_1darray_v4f16Dv4_DF16_iiiu18__amdgpu_texture_t(
// LLVM: call void @llvm.amdgcn.image.store.mip.1darray.v4f16.i32.v8i32(<4 x half> {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, <8 x i32> {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ void test_image_store_mip_1darray_v4f16(half4 val, int x, int slice, int mip, __amdgpu_texture_t rsrc) {
  __builtin_amdgcn_image_store_mip_1darray_v4f16_i32(val, 15, x, slice, mip, rsrc, 120, 110);
}

// CIR-LABEL: @_Z26test_image_load_mip_2d_f32
// CIR: cir.call_llvm_intrinsic "amdgcn.image.load.mip.2d" {{.*}} : (!s32i, !s32i, !s32i, !s32i, !cir.vector<8 x !s32i>, !s32i, !s32i) -> !cir.float
// LLVM: define{{.*}} float @_Z26test_image_load_mip_2d_f32iiiu18__amdgpu_texture_t(
// LLVM: call {{.*}}float @llvm.amdgcn.image.load.mip.2d.f32.i32.v8i32(i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, <8 x i32> {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ float test_image_load_mip_2d_f32(int x, int y, int mip, __amdgpu_texture_t rsrc) {
  return __builtin_amdgcn_image_load_mip_2d_f32_i32(8, x, y, mip, rsrc, 120, 110);
}

// CIR-LABEL: @_Z28test_image_load_mip_2d_v4f32
// CIR: cir.call_llvm_intrinsic "amdgcn.image.load.mip.2d" {{.*}} : (!s32i, !s32i, !s32i, !s32i, !cir.vector<8 x !s32i>, !s32i, !s32i) -> !cir.vector<4 x !cir.float>
// LLVM: define{{.*}} <4 x float> @_Z28test_image_load_mip_2d_v4f32iiiu18__amdgpu_texture_t(
// LLVM: call {{.*}}<4 x float> @llvm.amdgcn.image.load.mip.2d.v4f32.i32.v8i32(i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, <8 x i32> {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ float4 test_image_load_mip_2d_v4f32(int x, int y, int mip, __amdgpu_texture_t rsrc) {
  return __builtin_amdgcn_image_load_mip_2d_v4f32_i32(15, x, y, mip, rsrc, 120, 110);
}

// CIR-LABEL: @_Z28test_image_load_mip_2d_v4f16
// CIR: cir.call_llvm_intrinsic "amdgcn.image.load.mip.2d" {{.*}} : (!s32i, !s32i, !s32i, !s32i, !cir.vector<8 x !s32i>, !s32i, !s32i) -> !cir.vector<4 x !cir.f16>
// LLVM: define{{.*}} <4 x half> @_Z28test_image_load_mip_2d_v4f16iiiu18__amdgpu_texture_t(
// LLVM: call {{.*}}<4 x half> @llvm.amdgcn.image.load.mip.2d.v4f16.i32.v8i32(i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, <8 x i32> {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ half4 test_image_load_mip_2d_v4f16(int x, int y, int mip, __amdgpu_texture_t rsrc) {
  return __builtin_amdgcn_image_load_mip_2d_v4f16_i32(15, x, y, mip, rsrc, 120, 110);
}

// CIR-LABEL: @_Z27test_image_store_mip_2d_f32
// CIR: cir.call_llvm_intrinsic "amdgcn.image.store.mip.2d" {{.*}} : (!cir.float, !s32i, !s32i, !s32i, !s32i, !cir.vector<8 x !s32i>, !s32i, !s32i) -> !void
// LLVM: define{{.*}} void @_Z27test_image_store_mip_2d_f32fiiiu18__amdgpu_texture_t(
// LLVM: call void @llvm.amdgcn.image.store.mip.2d.f32.i32.v8i32(float {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, <8 x i32> {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ void test_image_store_mip_2d_f32(float val, int x, int y, int mip, __amdgpu_texture_t rsrc) {
  __builtin_amdgcn_image_store_mip_2d_f32_i32(val, 15, x, y, mip, rsrc, 120, 110);
}

// CIR-LABEL: @_Z29test_image_store_mip_2d_v4f32
// CIR: cir.call_llvm_intrinsic "amdgcn.image.store.mip.2d" {{.*}} : (!cir.vector<4 x !cir.float>, !s32i, !s32i, !s32i, !s32i, !cir.vector<8 x !s32i>, !s32i, !s32i) -> !void
// LLVM: define{{.*}} void @_Z29test_image_store_mip_2d_v4f32Dv4_fiiiu18__amdgpu_texture_t(
// LLVM: call void @llvm.amdgcn.image.store.mip.2d.v4f32.i32.v8i32(<4 x float> {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, <8 x i32> {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ void test_image_store_mip_2d_v4f32(float4 val, int x, int y, int mip, __amdgpu_texture_t rsrc) {
  __builtin_amdgcn_image_store_mip_2d_v4f32_i32(val, 15, x, y, mip, rsrc, 120, 110);
}

// CIR-LABEL: @_Z29test_image_store_mip_2d_v4f16
// CIR: cir.call_llvm_intrinsic "amdgcn.image.store.mip.2d" {{.*}} : (!cir.vector<4 x !cir.f16>, !s32i, !s32i, !s32i, !s32i, !cir.vector<8 x !s32i>, !s32i, !s32i) -> !void
// LLVM: define{{.*}} void @_Z29test_image_store_mip_2d_v4f16Dv4_DF16_iiiu18__amdgpu_texture_t(
// LLVM: call void @llvm.amdgcn.image.store.mip.2d.v4f16.i32.v8i32(<4 x half> {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, <8 x i32> {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ void test_image_store_mip_2d_v4f16(half4 val, int x, int y, int mip, __amdgpu_texture_t rsrc) {
  __builtin_amdgcn_image_store_mip_2d_v4f16_i32(val, 15, x, y, mip, rsrc, 120, 110);
}

// CIR-LABEL: @_Z31test_image_load_mip_2darray_f32
// CIR: cir.call_llvm_intrinsic "amdgcn.image.load.mip.2darray" {{.*}} : (!s32i, !s32i, !s32i, !s32i, !s32i, !cir.vector<8 x !s32i>, !s32i, !s32i) -> !cir.float
// LLVM: define{{.*}} float @_Z31test_image_load_mip_2darray_f32iiiiu18__amdgpu_texture_t(
// LLVM: call {{.*}}float @llvm.amdgcn.image.load.mip.2darray.f32.i32.v8i32(i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, <8 x i32> {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ float test_image_load_mip_2darray_f32(int x, int y, int slice, int mip, __amdgpu_texture_t rsrc) {
  return __builtin_amdgcn_image_load_mip_2darray_f32_i32(8, x, y, slice, mip, rsrc, 120, 110);
}

// CIR-LABEL: @_Z33test_image_load_mip_2darray_v4f32
// CIR: cir.call_llvm_intrinsic "amdgcn.image.load.mip.2darray" {{.*}} : (!s32i, !s32i, !s32i, !s32i, !s32i, !cir.vector<8 x !s32i>, !s32i, !s32i) -> !cir.vector<4 x !cir.float>
// LLVM: define{{.*}} <4 x float> @_Z33test_image_load_mip_2darray_v4f32iiiiu18__amdgpu_texture_t(
// LLVM: call {{.*}}<4 x float> @llvm.amdgcn.image.load.mip.2darray.v4f32.i32.v8i32(i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, <8 x i32> {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ float4 test_image_load_mip_2darray_v4f32(int x, int y, int slice, int mip, __amdgpu_texture_t rsrc) {
  return __builtin_amdgcn_image_load_mip_2darray_v4f32_i32(15, x, y, slice, mip, rsrc, 120, 110);
}

// CIR-LABEL: @_Z33test_image_load_mip_2darray_v4f16
// CIR: cir.call_llvm_intrinsic "amdgcn.image.load.mip.2darray" {{.*}} : (!s32i, !s32i, !s32i, !s32i, !s32i, !cir.vector<8 x !s32i>, !s32i, !s32i) -> !cir.vector<4 x !cir.f16>
// LLVM: define{{.*}} <4 x half> @_Z33test_image_load_mip_2darray_v4f16iiiiu18__amdgpu_texture_t(
// LLVM: call {{.*}}<4 x half> @llvm.amdgcn.image.load.mip.2darray.v4f16.i32.v8i32(i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, <8 x i32> {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ half4 test_image_load_mip_2darray_v4f16(int x, int y, int slice, int mip, __amdgpu_texture_t rsrc) {
  return __builtin_amdgcn_image_load_mip_2darray_v4f16_i32(15, x, y, slice, mip, rsrc, 120, 110);
}

// CIR-LABEL: @_Z32test_image_store_mip_2darray_f32
// CIR: cir.call_llvm_intrinsic "amdgcn.image.store.mip.2darray" {{.*}} : (!cir.float, !s32i, !s32i, !s32i, !s32i, !s32i, !cir.vector<8 x !s32i>, !s32i, !s32i) -> !void
// LLVM: define{{.*}} void @_Z32test_image_store_mip_2darray_f32fiiiiu18__amdgpu_texture_t(
// LLVM: call void @llvm.amdgcn.image.store.mip.2darray.f32.i32.v8i32(float {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, <8 x i32> {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ void test_image_store_mip_2darray_f32(float val, int x, int y, int slice, int mip, __amdgpu_texture_t rsrc) {
  __builtin_amdgcn_image_store_mip_2darray_f32_i32(val, 15, x, y, slice, mip, rsrc, 120, 110);
}

// CIR-LABEL: @_Z34test_image_store_mip_2darray_v4f32
// CIR: cir.call_llvm_intrinsic "amdgcn.image.store.mip.2darray" {{.*}} : (!cir.vector<4 x !cir.float>, !s32i, !s32i, !s32i, !s32i, !s32i, !cir.vector<8 x !s32i>, !s32i, !s32i) -> !void
// LLVM: define{{.*}} void @_Z34test_image_store_mip_2darray_v4f32Dv4_fiiiiu18__amdgpu_texture_t(
// LLVM: call void @llvm.amdgcn.image.store.mip.2darray.v4f32.i32.v8i32(<4 x float> {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, <8 x i32> {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ void test_image_store_mip_2darray_v4f32(float4 val, int x, int y, int slice, int mip, __amdgpu_texture_t rsrc) {
  __builtin_amdgcn_image_store_mip_2darray_v4f32_i32(val, 15, x, y, slice, mip, rsrc, 120, 110);
}

// CIR-LABEL: @_Z34test_image_store_mip_2darray_v4f16
// CIR: cir.call_llvm_intrinsic "amdgcn.image.store.mip.2darray" {{.*}} : (!cir.vector<4 x !cir.f16>, !s32i, !s32i, !s32i, !s32i, !s32i, !cir.vector<8 x !s32i>, !s32i, !s32i) -> !void
// LLVM: define{{.*}} void @_Z34test_image_store_mip_2darray_v4f16Dv4_DF16_iiiiu18__amdgpu_texture_t(
// LLVM: call void @llvm.amdgcn.image.store.mip.2darray.v4f16.i32.v8i32(<4 x half> {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, <8 x i32> {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ void test_image_store_mip_2darray_v4f16(half4 val, int x, int y, int slice, int mip, __amdgpu_texture_t rsrc) {
  __builtin_amdgcn_image_store_mip_2darray_v4f16_i32(val, 15, x, y, slice, mip, rsrc, 120, 110);
}

// CIR-LABEL: @_Z28test_image_load_mip_3d_v4f32
// CIR: cir.call_llvm_intrinsic "amdgcn.image.load.mip.3d" {{.*}} : (!s32i, !s32i, !s32i, !s32i, !s32i, !cir.vector<8 x !s32i>, !s32i, !s32i) -> !cir.vector<4 x !cir.float>
// LLVM: define{{.*}} <4 x float> @_Z28test_image_load_mip_3d_v4f32iiiiu18__amdgpu_texture_t(
// LLVM: call {{.*}}<4 x float> @llvm.amdgcn.image.load.mip.3d.v4f32.i32.v8i32(i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, <8 x i32> {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ float4 test_image_load_mip_3d_v4f32(int x, int y, int z, int mip, __amdgpu_texture_t rsrc) {
  return __builtin_amdgcn_image_load_mip_3d_v4f32_i32(15, x, y, z, mip, rsrc, 120, 110);
}

// CIR-LABEL: @_Z28test_image_load_mip_3d_v4f16
// CIR: cir.call_llvm_intrinsic "amdgcn.image.load.mip.3d" {{.*}} : (!s32i, !s32i, !s32i, !s32i, !s32i, !cir.vector<8 x !s32i>, !s32i, !s32i) -> !cir.vector<4 x !cir.f16>
// LLVM: define{{.*}} <4 x half> @_Z28test_image_load_mip_3d_v4f16iiiiu18__amdgpu_texture_t(
// LLVM: call {{.*}}<4 x half> @llvm.amdgcn.image.load.mip.3d.v4f16.i32.v8i32(i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, <8 x i32> {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ half4 test_image_load_mip_3d_v4f16(int x, int y, int z, int mip, __amdgpu_texture_t rsrc) {
  return __builtin_amdgcn_image_load_mip_3d_v4f16_i32(15, x, y, z, mip, rsrc, 120, 110);
}

// CIR-LABEL: @_Z29test_image_store_mip_3d_v4f32
// CIR: cir.call_llvm_intrinsic "amdgcn.image.store.mip.3d" {{.*}} : (!cir.vector<4 x !cir.float>, !s32i, !s32i, !s32i, !s32i, !s32i, !cir.vector<8 x !s32i>, !s32i, !s32i) -> !void
// LLVM: define{{.*}} void @_Z29test_image_store_mip_3d_v4f32Dv4_fiiiiu18__amdgpu_texture_t(
// LLVM: call void @llvm.amdgcn.image.store.mip.3d.v4f32.i32.v8i32(<4 x float> {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, <8 x i32> {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ void test_image_store_mip_3d_v4f32(float4 val, int x, int y, int z, int mip, __amdgpu_texture_t rsrc) {
  __builtin_amdgcn_image_store_mip_3d_v4f32_i32(val, 15, x, y, z, mip, rsrc, 120, 110);
}

// CIR-LABEL: @_Z29test_image_store_mip_3d_v4f16
// CIR: cir.call_llvm_intrinsic "amdgcn.image.store.mip.3d" {{.*}} : (!cir.vector<4 x !cir.f16>, !s32i, !s32i, !s32i, !s32i, !s32i, !cir.vector<8 x !s32i>, !s32i, !s32i) -> !void
// LLVM: define{{.*}} void @_Z29test_image_store_mip_3d_v4f16Dv4_DF16_iiiiu18__amdgpu_texture_t(
// LLVM: call void @llvm.amdgcn.image.store.mip.3d.v4f16.i32.v8i32(<4 x half> {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, <8 x i32> {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ void test_image_store_mip_3d_v4f16(half4 val, int x, int y, int z, int mip, __amdgpu_texture_t rsrc) {
  __builtin_amdgcn_image_store_mip_3d_v4f16_i32(val, 15, x, y, z, mip, rsrc, 120, 110);
}

// CIR-LABEL: @_Z30test_image_load_mip_cube_v4f32
// CIR: cir.call_llvm_intrinsic "amdgcn.image.load.mip.cube" {{.*}} : (!s32i, !s32i, !s32i, !s32i, !s32i, !cir.vector<8 x !s32i>, !s32i, !s32i) -> !cir.vector<4 x !cir.float>
// LLVM: define{{.*}} <4 x float> @_Z30test_image_load_mip_cube_v4f32iiiiu18__amdgpu_texture_t(
// LLVM: call {{.*}}<4 x float> @llvm.amdgcn.image.load.mip.cube.v4f32.i32.v8i32(i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, <8 x i32> {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ float4 test_image_load_mip_cube_v4f32(int x, int y, int face, int mip, __amdgpu_texture_t rsrc) {
  return __builtin_amdgcn_image_load_mip_cube_v4f32_i32(15, x, y, face, mip, rsrc, 120, 110);
}

// CIR-LABEL: @_Z30test_image_load_mip_cube_v4f16
// CIR: cir.call_llvm_intrinsic "amdgcn.image.load.mip.cube" {{.*}} : (!s32i, !s32i, !s32i, !s32i, !s32i, !cir.vector<8 x !s32i>, !s32i, !s32i) -> !cir.vector<4 x !cir.f16>
// LLVM: define{{.*}} <4 x half> @_Z30test_image_load_mip_cube_v4f16iiiiu18__amdgpu_texture_t(
// LLVM: call {{.*}}<4 x half> @llvm.amdgcn.image.load.mip.cube.v4f16.i32.v8i32(i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, <8 x i32> {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ half4 test_image_load_mip_cube_v4f16(int x, int y, int face, int mip, __amdgpu_texture_t rsrc) {
  return __builtin_amdgcn_image_load_mip_cube_v4f16_i32(15, x, y, face, mip, rsrc, 120, 110);
}

// CIR-LABEL: @_Z31test_image_store_mip_cube_v4f32
// CIR: cir.call_llvm_intrinsic "amdgcn.image.store.mip.cube" {{.*}} : (!cir.vector<4 x !cir.float>, !s32i, !s32i, !s32i, !s32i, !s32i, !cir.vector<8 x !s32i>, !s32i, !s32i) -> !void
// LLVM: define{{.*}} void @_Z31test_image_store_mip_cube_v4f32Dv4_fiiiiu18__amdgpu_texture_t(
// LLVM: call void @llvm.amdgcn.image.store.mip.cube.v4f32.i32.v8i32(<4 x float> {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, <8 x i32> {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ void test_image_store_mip_cube_v4f32(float4 val, int x, int y, int face, int mip, __amdgpu_texture_t rsrc) {
  __builtin_amdgcn_image_store_mip_cube_v4f32_i32(val, 15, x, y, face, mip, rsrc, 120, 110);
}

// CIR-LABEL: @_Z31test_image_store_mip_cube_v4f16
// CIR: cir.call_llvm_intrinsic "amdgcn.image.store.mip.cube" {{.*}} : (!cir.vector<4 x !cir.f16>, !s32i, !s32i, !s32i, !s32i, !s32i, !cir.vector<8 x !s32i>, !s32i, !s32i) -> !void
// LLVM: define{{.*}} void @_Z31test_image_store_mip_cube_v4f16Dv4_DF16_iiiiu18__amdgpu_texture_t(
// LLVM: call void @llvm.amdgcn.image.store.mip.cube.v4f16.i32.v8i32(<4 x half> {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, <8 x i32> {{.*}}, i32 {{.*}}, i32 {{.*}})
__device__ void test_image_store_mip_cube_v4f16(half4 val, int x, int y, int face, int mip, __amdgpu_texture_t rsrc) {
  __builtin_amdgcn_image_store_mip_cube_v4f16_i32(val, 15, x, y, face, mip, rsrc, 120, 110);
}
