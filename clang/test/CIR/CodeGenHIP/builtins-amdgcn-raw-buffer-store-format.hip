// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgpu8.03-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -fcuda-is-device -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple amdgpu8.03-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgpu8.03-amd-amdhsa -x hip -std=c++11 \
// RUN:            -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

#define __device__ __attribute__((device))

typedef float v4f32 __attribute__((ext_vector_type(4)));
typedef _Float16 v4f16 __attribute__((ext_vector_type(4)));

// CIR-LABEL: @_Z34test_raw_buffer_store_format_v4f32Dv4_fu22__amdgpu_buffer_rsrc_tii
// CIR: cir.call_llvm_intrinsic "amdgcn.raw.ptr.buffer.store.format" {{.*}} : (!cir.vector<4 x !cir.float>, !cir.ptr<!void, target_address_space(8)>, !s32i, !s32i, !s32i) -> !void
// LLVM: define{{.*}} void @_Z34test_raw_buffer_store_format_v4f32Dv4_fu22__amdgpu_buffer_rsrc_tii
// LLVM: call{{.*}} void @llvm.amdgcn.raw.ptr.buffer.store.format.v4f32(<4 x float> {{.*}})
__device__ void test_raw_buffer_store_format_v4f32(v4f32 vdata, __amdgpu_buffer_rsrc_t rsrc, int offset,
                                                     int soffset) {
  __builtin_amdgcn_raw_buffer_store_format_v4f32(vdata, rsrc, 0, 0, 0);
}

// CIR-LABEL: @_Z34test_raw_buffer_store_format_v4f16Dv4_DF16_u22__amdgpu_buffer_rsrc_tii
// CIR: cir.call_llvm_intrinsic "amdgcn.raw.ptr.buffer.store.format" {{.*}} : (!cir.vector<4 x !cir.f16>, !cir.ptr<!void, target_address_space(8)>, !s32i, !s32i, !s32i) -> !void
// LLVM: define{{.*}} void @_Z34test_raw_buffer_store_format_v4f16Dv4_DF16_u22__amdgpu_buffer_rsrc_tii
// LLVM: call{{.*}} void @llvm.amdgcn.raw.ptr.buffer.store.format.v4f16(<4 x half> {{.*}})
__device__ void test_raw_buffer_store_format_v4f16(v4f16 vdata, __amdgpu_buffer_rsrc_t rsrc, int offset,
                                                     int soffset) {
  __builtin_amdgcn_raw_buffer_store_format_v4f16(vdata, rsrc, 0, 0, 0);
}

// CIR-LABEL: @_Z51test_raw_buffer_store_format_v4f32_non_const_offsetDv4_fu22__amdgpu_buffer_rsrc_tii
// CIR: cir.call_llvm_intrinsic "amdgcn.raw.ptr.buffer.store.format" {{.*}} : (!cir.vector<4 x !cir.float>, !cir.ptr<!void, target_address_space(8)>, !s32i, !s32i, !s32i) -> !void
// LLVM: define{{.*}} void @_Z51test_raw_buffer_store_format_v4f32_non_const_offsetDv4_fu22__amdgpu_buffer_rsrc_tii
// LLVM: call{{.*}} void @llvm.amdgcn.raw.ptr.buffer.store.format.v4f32(<4 x float> {{.*}})
__device__ void test_raw_buffer_store_format_v4f32_non_const_offset(v4f32 vdata, __amdgpu_buffer_rsrc_t rsrc, int offset,
                                                                      int soffset) {
  __builtin_amdgcn_raw_buffer_store_format_v4f32(vdata, rsrc, offset, 0, 0);
}

// CIR-LABEL: @_Z52test_raw_buffer_store_format_v4f32_non_const_soffsetDv4_fu22__amdgpu_buffer_rsrc_tii
// CIR: cir.call_llvm_intrinsic "amdgcn.raw.ptr.buffer.store.format" {{.*}} : (!cir.vector<4 x !cir.float>, !cir.ptr<!void, target_address_space(8)>, !s32i, !s32i, !s32i) -> !void
// LLVM: define{{.*}} void @_Z52test_raw_buffer_store_format_v4f32_non_const_soffsetDv4_fu22__amdgpu_buffer_rsrc_tii
// LLVM: call{{.*}} void @llvm.amdgcn.raw.ptr.buffer.store.format.v4f32(<4 x float> {{.*}})
__device__ void test_raw_buffer_store_format_v4f32_non_const_soffset(v4f32 vdata, __amdgpu_buffer_rsrc_t rsrc, int offset,
                                                                       int soffset) {
  __builtin_amdgcn_raw_buffer_store_format_v4f32(vdata, rsrc, 0, soffset, 0);
}

// CIR-LABEL: @_Z33test_raw_buffer_load_format_v4f32u22__amdgpu_buffer_rsrc_tii
// CIR: cir.call_llvm_intrinsic "amdgcn.raw.ptr.buffer.load.format" {{.*}} : (!cir.ptr<!void, target_address_space(8)>, !s32i, !s32i, !s32i) -> !cir.vector<4 x !cir.float>
// LLVM: define{{.*}} <4 x float> @_Z33test_raw_buffer_load_format_v4f32u22__amdgpu_buffer_rsrc_tii
// LLVM: call{{.*}} <4 x float> @llvm.amdgcn.raw.ptr.buffer.load.format.v4f32({{.*}})
__device__ v4f32 test_raw_buffer_load_format_v4f32(__amdgpu_buffer_rsrc_t rsrc, int offset,
                                                     int soffset) {
  return __builtin_amdgcn_raw_buffer_load_format_v4f32(rsrc, 0, 0, 0);
}

// CIR-LABEL: @_Z33test_raw_buffer_load_format_v4f16u22__amdgpu_buffer_rsrc_tii
// CIR: cir.call_llvm_intrinsic "amdgcn.raw.ptr.buffer.load.format" {{.*}} : (!cir.ptr<!void, target_address_space(8)>, !s32i, !s32i, !s32i) -> !cir.vector<4 x !cir.f16>
// LLVM: define{{.*}} <4 x half> @_Z33test_raw_buffer_load_format_v4f16u22__amdgpu_buffer_rsrc_tii
// LLVM: call{{.*}} <4 x half> @llvm.amdgcn.raw.ptr.buffer.load.format.v4f16({{.*}})
__device__ v4f16 test_raw_buffer_load_format_v4f16(__amdgpu_buffer_rsrc_t rsrc, int offset,
                                                     int soffset) {
  return __builtin_amdgcn_raw_buffer_load_format_v4f16(rsrc, 0, 0, 0);
}

// CIR-LABEL: @_Z50test_raw_buffer_load_format_v4f32_non_const_offsetu22__amdgpu_buffer_rsrc_tii
// CIR: cir.call_llvm_intrinsic "amdgcn.raw.ptr.buffer.load.format" {{.*}} : (!cir.ptr<!void, target_address_space(8)>, !s32i, !s32i, !s32i) -> !cir.vector<4 x !cir.float>
// LLVM: define{{.*}} <4 x float> @_Z50test_raw_buffer_load_format_v4f32_non_const_offsetu22__amdgpu_buffer_rsrc_tii
// LLVM: call{{.*}} <4 x float> @llvm.amdgcn.raw.ptr.buffer.load.format.v4f32({{.*}})
__device__ v4f32 test_raw_buffer_load_format_v4f32_non_const_offset(__amdgpu_buffer_rsrc_t rsrc, int offset,
                                                                      int soffset) {
  return __builtin_amdgcn_raw_buffer_load_format_v4f32(rsrc, offset, 0, 0);
}

// CIR-LABEL: @_Z51test_raw_buffer_load_format_v4f32_non_const_soffsetu22__amdgpu_buffer_rsrc_tii
// CIR: cir.call_llvm_intrinsic "amdgcn.raw.ptr.buffer.load.format" {{.*}} : (!cir.ptr<!void, target_address_space(8)>, !s32i, !s32i, !s32i) -> !cir.vector<4 x !cir.float>
// LLVM: define{{.*}} <4 x float> @_Z51test_raw_buffer_load_format_v4f32_non_const_soffsetu22__amdgpu_buffer_rsrc_tii
// LLVM: call{{.*}} <4 x float> @llvm.amdgcn.raw.ptr.buffer.load.format.v4f32({{.*}})
__device__ v4f32 test_raw_buffer_load_format_v4f32_non_const_soffset(__amdgpu_buffer_rsrc_t rsrc, int offset,
                                                                       int soffset) {
  return __builtin_amdgcn_raw_buffer_load_format_v4f32(rsrc, 0, soffset, 0);
}
