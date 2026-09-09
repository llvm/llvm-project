// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgpu12.50-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN: -fcuda-is-device -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple amdgpu12.50-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN: -fcuda-is-device -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s

// RUN: %clang_cc1 -triple amdgpu12.50-amd-amdhsa -x hip -std=c++11 \
// RUN: -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// These intrinsics are overloaded on their result type, not on an argument, so
// the result type is passed to the emission helper explicitly.

#define __device__ __attribute__((device))
#define global __attribute__((address_space(1)))

typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v4i __attribute__((ext_vector_type(4)));

// CIR-LABEL: @_Z28test_amdgcn_cluster_load_b32PU3AS1ii
// CIR: cir.call_llvm_intrinsic "amdgcn.cluster.load.b32" {{.*}} : (!cir.ptr<!s32i, target_address_space(1)>, !s32i, !s32i) -> !s32i
// LLVM: define{{.*}} i32 @_Z28test_amdgcn_cluster_load_b32PU3AS1ii
// LLVM: call{{.*}} i32 @llvm.amdgcn.cluster.load.b32.i32(ptr addrspace(1) %{{.+}}, i32 10, i32 %{{.+}})
__device__ int test_amdgcn_cluster_load_b32(global int* inptr, int mask) {
  return __builtin_amdgcn_cluster_load_b32(inptr, 10, mask);
}

// CIR-LABEL: @_Z28test_amdgcn_cluster_load_b64PU3AS1Dv2_ii
// CIR: cir.call_llvm_intrinsic "amdgcn.cluster.load.b64" {{.*}} : (!cir.ptr<!cir.vector<2 x !s32i>, target_address_space(1)>, !s32i, !s32i) -> !cir.vector<2 x !s32i>
// LLVM: define{{.*}} <2 x i32> @_Z28test_amdgcn_cluster_load_b64PU3AS1Dv2_ii
// LLVM: call{{.*}} <2 x i32> @llvm.amdgcn.cluster.load.b64.v2i32(ptr addrspace(1) %{{.+}}, i32 22, i32 %{{.+}})
__device__ v2i test_amdgcn_cluster_load_b64(global v2i* inptr, int mask) {
  return __builtin_amdgcn_cluster_load_b64(inptr, 22, mask);
}

// CIR-LABEL: @_Z29test_amdgcn_cluster_load_b128PU3AS1Dv4_ii
// CIR: cir.call_llvm_intrinsic "amdgcn.cluster.load.b128" {{.*}} : (!cir.ptr<!cir.vector<4 x !s32i>, target_address_space(1)>, !s32i, !s32i) -> !cir.vector<4 x !s32i>
// LLVM: define{{.*}} <4 x i32> @_Z29test_amdgcn_cluster_load_b128PU3AS1Dv4_ii
// LLVM: call{{.*}} <4 x i32> @llvm.amdgcn.cluster.load.b128.v4i32(ptr addrspace(1) %{{.+}}, i32 27, i32 %{{.+}})
__device__ v4i test_amdgcn_cluster_load_b128(global v4i* inptr, int mask) {
  return __builtin_amdgcn_cluster_load_b128(inptr, 27, mask);
}
