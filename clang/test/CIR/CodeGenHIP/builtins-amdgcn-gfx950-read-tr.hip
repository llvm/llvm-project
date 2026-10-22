// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgpu9.50-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN: -fcuda-is-device -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple amdgpu9.50-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN: -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgpu9.50-amd-amdhsa -x hip -std=c++11 \
// RUN: -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// FIXME: CIR doesn't propagate the 'contract' fast-math flag to LLVM IR
// calls yet, so the LLVM check lines use call{{.*}} to tolerate the
// difference between CIR (no flags) and classic codegen ('contract').

#define __device__ __attribute__((device))

typedef int    v2i __attribute__((ext_vector_type(2)));
typedef int    v3i __attribute__((ext_vector_type(3)));
typedef __fp16 v4h __attribute__((ext_vector_type(4)));
typedef short  v4s __attribute__((ext_vector_type(4)));
typedef __bf16 v4y __attribute__((ext_vector_type(4)));

typedef __attribute__((address_space(3))) v2i *v2i_as3_ptr;
typedef __attribute__((address_space(3))) v3i *v3i_as3_ptr;
typedef __attribute__((address_space(3))) v4h *v4h_as3_ptr;
typedef __attribute__((address_space(3))) v4s *v4s_as3_ptr;
typedef __attribute__((address_space(3))) v4y *v4y_as3_ptr;

// CIR-LABEL: @_Z26test_ds_read_tr4_b64_v2i32PU3AS3Dv2_i
// CIR: cir.call_llvm_intrinsic "amdgcn.ds.read.tr4.b64" {{.*}} : (!cir.ptr<!cir.vector<2 x !s32i>, target_address_space(3)>) -> !cir.vector<2 x !s32i>
// LLVM-LABEL: @_Z26test_ds_read_tr4_b64_v2i32PU3AS3Dv2_i
// LLVM: call{{.*}} <2 x i32> @llvm.amdgcn.ds.read.tr4.b64.v2i32(ptr addrspace(3) %{{.*}})
__device__ v2i test_ds_read_tr4_b64_v2i32(v2i_as3_ptr inptr) {
  return __builtin_amdgcn_ds_read_tr4_b64_v2i32(inptr);
}

// CIR-LABEL: @_Z26test_ds_read_tr6_b96_v3i32PU3AS3Dv3_i
// CIR: cir.call_llvm_intrinsic "amdgcn.ds.read.tr6.b96" {{.*}} : (!cir.ptr<!cir.vector<3 x !s32i>, target_address_space(3)>) -> !cir.vector<3 x !s32i>
// LLVM-LABEL: @_Z26test_ds_read_tr6_b96_v3i32PU3AS3Dv3_i
// LLVM: call{{.*}} <3 x i32> @llvm.amdgcn.ds.read.tr6.b96.v3i32(ptr addrspace(3) %{{.*}})
__device__ v3i test_ds_read_tr6_b96_v3i32(v3i_as3_ptr inptr) {
  return __builtin_amdgcn_ds_read_tr6_b96_v3i32(inptr);
}

// CIR-LABEL: @_Z26test_ds_read_tr8_b64_v2i32PU3AS3Dv2_i
// CIR: cir.call_llvm_intrinsic "amdgcn.ds.read.tr8.b64" {{.*}} : (!cir.ptr<!cir.vector<2 x !s32i>, target_address_space(3)>) -> !cir.vector<2 x !s32i>
// LLVM-LABEL: @_Z26test_ds_read_tr8_b64_v2i32PU3AS3Dv2_i
// LLVM: call{{.*}} <2 x i32> @llvm.amdgcn.ds.read.tr8.b64.v2i32(ptr addrspace(3) %{{.*}})
__device__ v2i test_ds_read_tr8_b64_v2i32(v2i_as3_ptr inptr) {
  return __builtin_amdgcn_ds_read_tr8_b64_v2i32(inptr);
}

// CIR-LABEL: @_Z27test_ds_read_tr16_b64_v4i16PU3AS3Dv4_s
// CIR: cir.call_llvm_intrinsic "amdgcn.ds.read.tr16.b64" {{.*}} : (!cir.ptr<!cir.vector<4 x !s16i>, target_address_space(3)>) -> !cir.vector<4 x !s16i>
// LLVM-LABEL: @_Z27test_ds_read_tr16_b64_v4i16PU3AS3Dv4_s
// LLVM: call{{.*}} <4 x i16> @llvm.amdgcn.ds.read.tr16.b64.v4i16(ptr addrspace(3) %{{.*}})
__device__ v4s test_ds_read_tr16_b64_v4i16(v4s_as3_ptr inptr) {
  return __builtin_amdgcn_ds_read_tr16_b64_v4i16(inptr);
}

// CIR-LABEL: @_Z27test_ds_read_tr16_b64_v4f16PU3AS3Dv4_Dh
// CIR: cir.call_llvm_intrinsic "amdgcn.ds.read.tr16.b64" {{.*}} : (!cir.ptr<!cir.vector<4 x !cir.f16>, target_address_space(3)>) -> !cir.vector<4 x !cir.f16>
// LLVM-LABEL: @_Z27test_ds_read_tr16_b64_v4f16PU3AS3Dv4_Dh
// LLVM: call{{.*}} <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) %{{.*}})
__device__ v4h test_ds_read_tr16_b64_v4f16(v4h_as3_ptr inptr) {
  return __builtin_amdgcn_ds_read_tr16_b64_v4f16(inptr);
}

// CIR-LABEL: @_Z28test_ds_read_tr16_b64_v4bf16PU3AS3Dv4_DF16b
// CIR: cir.call_llvm_intrinsic "amdgcn.ds.read.tr16.b64" {{.*}} : (!cir.ptr<!cir.vector<4 x !cir.bf16>, target_address_space(3)>) -> !cir.vector<4 x !cir.bf16>
// LLVM-LABEL: @_Z28test_ds_read_tr16_b64_v4bf16PU3AS3Dv4_DF16b
// LLVM: call{{.*}} <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) %{{.*}})
__device__ v4y test_ds_read_tr16_b64_v4bf16(v4y_as3_ptr inptr) {
  return __builtin_amdgcn_ds_read_tr16_b64_v4bf16(inptr);
}
