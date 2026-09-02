// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgpu12.00-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN: -fcuda-is-device -target-feature +wavefrontsize64 -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple amdgpu12.00-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN: -fcuda-is-device -target-feature +wavefrontsize64 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgpu12.00-amd-amdhsa -x hip -std=c++11 \
// RUN: -fcuda-is-device -target-feature +wavefrontsize64 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// FIXME: CIR doesn't propagate the 'contract' fast-math flag to LLVM IR
// calls yet, so the LLVM check lines use call{{.*}} to tolerate the
// difference between CIR (no flags) and classic codegen ('contract').

#define __device__ __attribute__((device))

typedef __fp16 v4h __attribute__((ext_vector_type(4)));
typedef short  v4s __attribute__((ext_vector_type(4)));
typedef __bf16 v4y __attribute__((ext_vector_type(4)));

typedef __attribute__((address_space(1))) int *int_as1_ptr;
typedef __attribute__((address_space(1))) v4h *v4h_as1_ptr;
typedef __attribute__((address_space(1))) v4s *v4s_as1_ptr;
typedef __attribute__((address_space(1))) v4y *v4y_as1_ptr;

// CIR-LABEL: @_Z27test_global_load_tr_b64_i32PU3AS1i
// CIR: cir.call_llvm_intrinsic "amdgcn.global.load.tr.b64" {{.*}} : (!cir.ptr<!s32i, target_address_space(1)>) -> !s32i
// LLVM-LABEL: @_Z27test_global_load_tr_b64_i32PU3AS1i
// LLVM: call{{.*}} i32 @llvm.amdgcn.global.load.tr.b64.i32(ptr addrspace(1) %{{.*}})
__device__ int test_global_load_tr_b64_i32(int_as1_ptr inptr) {
  return __builtin_amdgcn_global_load_tr_b64_i32(inptr);
}

// CIR-LABEL: @_Z30test_global_load_tr_b128_v4i16PU3AS1Dv4_s
// CIR: cir.call_llvm_intrinsic "amdgcn.global.load.tr.b128" {{.*}} : (!cir.ptr<!cir.vector<4 x !s16i>, target_address_space(1)>) -> !cir.vector<4 x !s16i>
// LLVM-LABEL: @_Z30test_global_load_tr_b128_v4i16PU3AS1Dv4_s
// LLVM: call{{.*}} <4 x i16> @llvm.amdgcn.global.load.tr.b128.v4i16(ptr addrspace(1) %{{.*}})
__device__ v4s test_global_load_tr_b128_v4i16(v4s_as1_ptr inptr) {
  return __builtin_amdgcn_global_load_tr_b128_v4i16(inptr);
}

// CIR-LABEL: @_Z30test_global_load_tr_b128_v4f16PU3AS1Dv4_Dh
// CIR: cir.call_llvm_intrinsic "amdgcn.global.load.tr.b128" {{.*}} : (!cir.ptr<!cir.vector<4 x !cir.f16>, target_address_space(1)>) -> !cir.vector<4 x !cir.f16>
// LLVM-LABEL: @_Z30test_global_load_tr_b128_v4f16PU3AS1Dv4_Dh
// LLVM: call{{.*}} <4 x half> @llvm.amdgcn.global.load.tr.b128.v4f16(ptr addrspace(1) %{{.*}})
__device__ v4h test_global_load_tr_b128_v4f16(v4h_as1_ptr inptr) {
  return __builtin_amdgcn_global_load_tr_b128_v4f16(inptr);
}

// CIR-LABEL: @_Z31test_global_load_tr_b128_v4bf16PU3AS1Dv4_DF16b
// CIR: cir.call_llvm_intrinsic "amdgcn.global.load.tr.b128" {{.*}} : (!cir.ptr<!cir.vector<4 x !cir.bf16>, target_address_space(1)>) -> !cir.vector<4 x !cir.bf16>
// LLVM-LABEL: @_Z31test_global_load_tr_b128_v4bf16PU3AS1Dv4_DF16b
// LLVM: call{{.*}} <4 x bfloat> @llvm.amdgcn.global.load.tr.b128.v4bf16(ptr addrspace(1) %{{.*}})
__device__ v4y test_global_load_tr_b128_v4bf16(v4y_as1_ptr inptr) {
  return __builtin_amdgcn_global_load_tr_b128_v4bf16(inptr);
}
