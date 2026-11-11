// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgpu12.00-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN: -fcuda-is-device -target-feature +wavefrontsize32 -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple amdgpu12.00-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN: -fcuda-is-device -target-feature +wavefrontsize32 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgpu12.00-amd-amdhsa -x hip -std=c++11 \
// RUN: -fcuda-is-device -target-feature +wavefrontsize32 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// FIXME: CIR doesn't propagate the 'contract' fast-math flag to LLVM IR
// calls yet, so the LLVM check lines use call{{.*}} to tolerate the
// difference between CIR (no flags) and classic codegen ('contract').

#define __device__ __attribute__((device))

typedef int    v2i __attribute__((ext_vector_type(2)));
typedef __fp16 v8h __attribute__((ext_vector_type(8)));
typedef short  v8s __attribute__((ext_vector_type(8)));
typedef __bf16 v8y __attribute__((ext_vector_type(8)));

typedef __attribute__((address_space(1))) v2i *v2i_as1_ptr;
typedef __attribute__((address_space(1))) v8h *v8h_as1_ptr;
typedef __attribute__((address_space(1))) v8s *v8s_as1_ptr;
typedef __attribute__((address_space(1))) v8y *v8y_as1_ptr;

// CIR-LABEL: @_Z29test_global_load_tr_b64_v2i32PU3AS1Dv2_i
// CIR: cir.call_llvm_intrinsic "amdgcn.global.load.tr.b64" {{.*}} : (!cir.ptr<!cir.vector<2 x !s32i>, target_address_space(1)>) -> !cir.vector<2 x !s32i>
// LLVM-LABEL: @_Z29test_global_load_tr_b64_v2i32PU3AS1Dv2_i
// LLVM: call{{.*}} <2 x i32> @llvm.amdgcn.global.load.tr.b64.v2i32(ptr addrspace(1) %{{.*}})
__device__ v2i test_global_load_tr_b64_v2i32(v2i_as1_ptr inptr) {
  return __builtin_amdgcn_global_load_tr_b64_v2i32(inptr);
}

// CIR-LABEL: @_Z30test_global_load_tr_b128_v8i16PU3AS1Dv8_s
// CIR: cir.call_llvm_intrinsic "amdgcn.global.load.tr.b128" {{.*}} : (!cir.ptr<!cir.vector<8 x !s16i>, target_address_space(1)>) -> !cir.vector<8 x !s16i>
// LLVM-LABEL: @_Z30test_global_load_tr_b128_v8i16PU3AS1Dv8_s
// LLVM: call{{.*}} <8 x i16> @llvm.amdgcn.global.load.tr.b128.v8i16(ptr addrspace(1) %{{.*}})
__device__ v8s test_global_load_tr_b128_v8i16(v8s_as1_ptr inptr) {
  return __builtin_amdgcn_global_load_tr_b128_v8i16(inptr);
}

// CIR-LABEL: @_Z30test_global_load_tr_b128_v8f16PU3AS1Dv8_Dh
// CIR: cir.call_llvm_intrinsic "amdgcn.global.load.tr.b128" {{.*}} : (!cir.ptr<!cir.vector<8 x !cir.f16>, target_address_space(1)>) -> !cir.vector<8 x !cir.f16>
// LLVM-LABEL: @_Z30test_global_load_tr_b128_v8f16PU3AS1Dv8_Dh
// LLVM: call{{.*}} <8 x half> @llvm.amdgcn.global.load.tr.b128.v8f16(ptr addrspace(1) %{{.*}})
__device__ v8h test_global_load_tr_b128_v8f16(v8h_as1_ptr inptr) {
  return __builtin_amdgcn_global_load_tr_b128_v8f16(inptr);
}

// CIR-LABEL: @_Z31test_global_load_tr_b128_v8bf16PU3AS1Dv8_DF16b
// CIR: cir.call_llvm_intrinsic "amdgcn.global.load.tr.b128" {{.*}} : (!cir.ptr<!cir.vector<8 x !cir.bf16>, target_address_space(1)>) -> !cir.vector<8 x !cir.bf16>
// LLVM-LABEL: @_Z31test_global_load_tr_b128_v8bf16PU3AS1Dv8_DF16b
// LLVM: call{{.*}} <8 x bfloat> @llvm.amdgcn.global.load.tr.b128.v8bf16(ptr addrspace(1) %{{.*}})
__device__ v8y test_global_load_tr_b128_v8bf16(v8y_as1_ptr inptr) {
  return __builtin_amdgcn_global_load_tr_b128_v8bf16(inptr);
}
