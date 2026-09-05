// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgpu12.50-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN: -fcuda-is-device -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple amdgpu12.50-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN: -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgpu12.50-amd-amdhsa -x hip -std=c++11 \
// RUN: -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// FIXME: CIR doesn't propagate the 'contract' fast-math flag to LLVM IR
// calls yet, so the LLVM check lines use call{{.*}} to tolerate the
// difference between CIR (no flags) and classic codegen ('contract').

#define __device__ __attribute__((device))

typedef int    v2i __attribute__((ext_vector_type(2)));
typedef int    v3i __attribute__((ext_vector_type(3)));
typedef __fp16 v8h __attribute__((ext_vector_type(8)));
typedef short  v8s __attribute__((ext_vector_type(8)));
typedef __bf16 v8y __attribute__((ext_vector_type(8)));

typedef __attribute__((address_space(1))) v2i *v2i_as1_ptr;
typedef __attribute__((address_space(3))) v2i *v2i_as3_ptr;
typedef __attribute__((address_space(1))) v3i *v3i_as1_ptr;
typedef __attribute__((address_space(3))) v3i *v3i_as3_ptr;
typedef __attribute__((address_space(1))) v8h *v8h_as1_ptr;
typedef __attribute__((address_space(3))) v8h *v8h_as3_ptr;
typedef __attribute__((address_space(1))) v8s *v8s_as1_ptr;
typedef __attribute__((address_space(3))) v8s *v8s_as3_ptr;
typedef __attribute__((address_space(1))) v8y *v8y_as1_ptr;
typedef __attribute__((address_space(3))) v8y *v8y_as3_ptr;

// CIR-LABEL: @_Z30test_global_load_tr4_b64_v2i32PU3AS1Dv2_i
// CIR: cir.call_llvm_intrinsic "amdgcn.global.load.tr4.b64" {{.*}} : (!cir.ptr<!cir.vector<2 x !s32i>, target_address_space(1)>) -> !cir.vector<2 x !s32i>
// LLVM-LABEL: @_Z30test_global_load_tr4_b64_v2i32PU3AS1Dv2_i
// LLVM: call{{.*}} <2 x i32> @llvm.amdgcn.global.load.tr4.b64.v2i32(ptr addrspace(1) %{{.*}})
__device__ v2i test_global_load_tr4_b64_v2i32(v2i_as1_ptr inptr) {
  return __builtin_amdgcn_global_load_tr4_b64_v2i32(inptr);
}

// CIR-LABEL: @_Z30test_global_load_tr8_b64_v2i32PU3AS1Dv2_i
// CIR: cir.call_llvm_intrinsic "amdgcn.global.load.tr.b64" {{.*}} : (!cir.ptr<!cir.vector<2 x !s32i>, target_address_space(1)>) -> !cir.vector<2 x !s32i>
// LLVM-LABEL: @_Z30test_global_load_tr8_b64_v2i32PU3AS1Dv2_i
// LLVM: call{{.*}} <2 x i32> @llvm.amdgcn.global.load.tr.b64.v2i32(ptr addrspace(1) %{{.*}})
__device__ v2i test_global_load_tr8_b64_v2i32(v2i_as1_ptr inptr) {
  return __builtin_amdgcn_global_load_tr8_b64_v2i32(inptr);
}

// CIR-LABEL: @_Z30test_global_load_tr6_b96_v3i32PU3AS1Dv3_i
// CIR: cir.call_llvm_intrinsic "amdgcn.global.load.tr6.b96" {{.*}} : (!cir.ptr<!cir.vector<3 x !s32i>, target_address_space(1)>) -> !cir.vector<3 x !s32i>
// LLVM-LABEL: @_Z30test_global_load_tr6_b96_v3i32PU3AS1Dv3_i
// LLVM: call{{.*}} <3 x i32> @llvm.amdgcn.global.load.tr6.b96.v3i32(ptr addrspace(1) %{{.*}})
__device__ v3i test_global_load_tr6_b96_v3i32(v3i_as1_ptr inptr) {
  return __builtin_amdgcn_global_load_tr6_b96_v3i32(inptr);
}

// CIR-LABEL: @_Z32test_global_load_tr16_b128_v8i16PU3AS1Dv8_s
// CIR: cir.call_llvm_intrinsic "amdgcn.global.load.tr.b128" {{.*}} : (!cir.ptr<!cir.vector<8 x !s16i>, target_address_space(1)>) -> !cir.vector<8 x !s16i>
// LLVM-LABEL: @_Z32test_global_load_tr16_b128_v8i16PU3AS1Dv8_s
// LLVM: call{{.*}} <8 x i16> @llvm.amdgcn.global.load.tr.b128.v8i16(ptr addrspace(1) %{{.*}})
__device__ v8s test_global_load_tr16_b128_v8i16(v8s_as1_ptr inptr) {
  return __builtin_amdgcn_global_load_tr16_b128_v8i16(inptr);
}

// CIR-LABEL: @_Z32test_global_load_tr16_b128_v8f16PU3AS1Dv8_Dh
// CIR: cir.call_llvm_intrinsic "amdgcn.global.load.tr.b128" {{.*}} : (!cir.ptr<!cir.vector<8 x !cir.f16>, target_address_space(1)>) -> !cir.vector<8 x !cir.f16>
// LLVM-LABEL: @_Z32test_global_load_tr16_b128_v8f16PU3AS1Dv8_Dh
// LLVM: call{{.*}} <8 x half> @llvm.amdgcn.global.load.tr.b128.v8f16(ptr addrspace(1) %{{.*}})
__device__ v8h test_global_load_tr16_b128_v8f16(v8h_as1_ptr inptr) {
  return __builtin_amdgcn_global_load_tr16_b128_v8f16(inptr);
}

// CIR-LABEL: @_Z33test_global_load_tr16_b128_v8bf16PU3AS1Dv8_DF16b
// CIR: cir.call_llvm_intrinsic "amdgcn.global.load.tr.b128" {{.*}} : (!cir.ptr<!cir.vector<8 x !cir.bf16>, target_address_space(1)>) -> !cir.vector<8 x !cir.bf16>
// LLVM-LABEL: @_Z33test_global_load_tr16_b128_v8bf16PU3AS1Dv8_DF16b
// LLVM: call{{.*}} <8 x bfloat> @llvm.amdgcn.global.load.tr.b128.v8bf16(ptr addrspace(1) %{{.*}})
__device__ v8y test_global_load_tr16_b128_v8bf16(v8y_as1_ptr inptr) {
  return __builtin_amdgcn_global_load_tr16_b128_v8bf16(inptr);
}

// CIR-LABEL: @_Z26test_ds_load_tr4_b64_v2i32PU3AS3Dv2_i
// CIR: cir.call_llvm_intrinsic "amdgcn.ds.load.tr4.b64" {{.*}} : (!cir.ptr<!cir.vector<2 x !s32i>, target_address_space(3)>) -> !cir.vector<2 x !s32i>
// LLVM-LABEL: @_Z26test_ds_load_tr4_b64_v2i32PU3AS3Dv2_i
// LLVM: call{{.*}} <2 x i32> @llvm.amdgcn.ds.load.tr4.b64.v2i32(ptr addrspace(3) %{{.*}})
__device__ v2i test_ds_load_tr4_b64_v2i32(v2i_as3_ptr inptr) {
  return __builtin_amdgcn_ds_load_tr4_b64_v2i32(inptr);
}

// CIR-LABEL: @_Z26test_ds_load_tr8_b64_v2i32PU3AS3Dv2_i
// CIR: cir.call_llvm_intrinsic "amdgcn.ds.load.tr8.b64" {{.*}} : (!cir.ptr<!cir.vector<2 x !s32i>, target_address_space(3)>) -> !cir.vector<2 x !s32i>
// LLVM-LABEL: @_Z26test_ds_load_tr8_b64_v2i32PU3AS3Dv2_i
// LLVM: call{{.*}} <2 x i32> @llvm.amdgcn.ds.load.tr8.b64.v2i32(ptr addrspace(3) %{{.*}})
__device__ v2i test_ds_load_tr8_b64_v2i32(v2i_as3_ptr inptr) {
  return __builtin_amdgcn_ds_load_tr8_b64_v2i32(inptr);
}

// CIR-LABEL: @_Z26test_ds_load_tr6_b96_v3i32PU3AS3Dv3_i
// CIR: cir.call_llvm_intrinsic "amdgcn.ds.load.tr6.b96" {{.*}} : (!cir.ptr<!cir.vector<3 x !s32i>, target_address_space(3)>) -> !cir.vector<3 x !s32i>
// LLVM-LABEL: @_Z26test_ds_load_tr6_b96_v3i32PU3AS3Dv3_i
// LLVM: call{{.*}} <3 x i32> @llvm.amdgcn.ds.load.tr6.b96.v3i32(ptr addrspace(3) %{{.*}})
__device__ v3i test_ds_load_tr6_b96_v3i32(v3i_as3_ptr inptr) {
  return __builtin_amdgcn_ds_load_tr6_b96_v3i32(inptr);
}

// CIR-LABEL: @_Z28test_ds_load_tr16_b128_v8i16PU3AS3Dv8_s
// CIR: cir.call_llvm_intrinsic "amdgcn.ds.load.tr16.b128" {{.*}} : (!cir.ptr<!cir.vector<8 x !s16i>, target_address_space(3)>) -> !cir.vector<8 x !s16i>
// LLVM-LABEL: @_Z28test_ds_load_tr16_b128_v8i16PU3AS3Dv8_s
// LLVM: call{{.*}} <8 x i16> @llvm.amdgcn.ds.load.tr16.b128.v8i16(ptr addrspace(3) %{{.*}})
__device__ v8s test_ds_load_tr16_b128_v8i16(v8s_as3_ptr inptr) {
  return __builtin_amdgcn_ds_load_tr16_b128_v8i16(inptr);
}

// CIR-LABEL: @_Z28test_ds_load_tr16_b128_v8f16PU3AS3Dv8_Dh
// CIR: cir.call_llvm_intrinsic "amdgcn.ds.load.tr16.b128" {{.*}} : (!cir.ptr<!cir.vector<8 x !cir.f16>, target_address_space(3)>) -> !cir.vector<8 x !cir.f16>
// LLVM-LABEL: @_Z28test_ds_load_tr16_b128_v8f16PU3AS3Dv8_Dh
// LLVM: call{{.*}} <8 x half> @llvm.amdgcn.ds.load.tr16.b128.v8f16(ptr addrspace(3) %{{.*}})
__device__ v8h test_ds_load_tr16_b128_v8f16(v8h_as3_ptr inptr) {
  return __builtin_amdgcn_ds_load_tr16_b128_v8f16(inptr);
}

// CIR-LABEL: @_Z29test_ds_load_tr16_b128_v8bf16PU3AS3Dv8_DF16b
// CIR: cir.call_llvm_intrinsic "amdgcn.ds.load.tr16.b128" {{.*}} : (!cir.ptr<!cir.vector<8 x !cir.bf16>, target_address_space(3)>) -> !cir.vector<8 x !cir.bf16>
// LLVM-LABEL: @_Z29test_ds_load_tr16_b128_v8bf16PU3AS3Dv8_DF16b
// LLVM: call{{.*}} <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %{{.*}})
__device__ v8y test_ds_load_tr16_b128_v8bf16(v8y_as3_ptr inptr) {
  return __builtin_amdgcn_ds_load_tr16_b128_v8bf16(inptr);
}
