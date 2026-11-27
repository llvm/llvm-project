// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgpu12.00-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -fcuda-is-device -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple amdgpu12.00-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgpu12.00-amd-amdhsa -x hip -std=c++11 \
// RUN:            -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

#define __device__ __attribute__((device))

typedef __attribute__((address_space(1))) float *global_ptr_t;
typedef __attribute__((address_space(4))) const char *constant_ptr_t;

//===----------------------------------------------------------------------===//
// Test AMDGPU builtins
//===----------------------------------------------------------------------===//

// CIR-LABEL: @_Z20test_s_prefetch_dataPiPU3AS1fPU3AS4Kcj
// CIR: cir.call_llvm_intrinsic "amdgcn.s.prefetch.data" {{.*}} : (!cir.ptr<!void>, !u32i) -> !void
// CIR: cir.call_llvm_intrinsic "amdgcn.s.prefetch.data" {{.*}} : (!cir.ptr<!void, target_address_space(1)>, !u32i) -> !void
// CIR: cir.call_llvm_intrinsic "amdgcn.s.prefetch.data" {{.*}} : (!cir.ptr<!void, target_address_space(4)>, !u32i) -> !void
// LLVM: define{{.*}} void @_Z20test_s_prefetch_dataPiPU3AS1fPU3AS4Kcj
// LLVM: call void @llvm.amdgcn.s.prefetch.data.p0(ptr %{{.*}}, i32 0)
// LLVM: call void @llvm.amdgcn.s.prefetch.data.p1(ptr addrspace(1) %{{.*}}, i32 %{{.*}})
// LLVM: call void @llvm.amdgcn.s.prefetch.data.p4(ptr addrspace(4) %{{.*}}, i32 31)
__device__ void test_s_prefetch_data(int *fp, global_ptr_t gp,
                                     constant_ptr_t cp, unsigned int len) {
  __builtin_amdgcn_s_prefetch_data(fp, 0);
  __builtin_amdgcn_s_prefetch_data(gp, len);
  __builtin_amdgcn_s_prefetch_data(cp, 31);
}

// CIR-LABEL: @_Z20test_s_prefetch_instPiPU3AS1fPU3AS4Kcj
// CIR: cir.call_llvm_intrinsic "amdgcn.s.prefetch.inst" {{.*}} : (!cir.ptr<!void>, !u32i) -> !void
// CIR: cir.call_llvm_intrinsic "amdgcn.s.prefetch.inst" {{.*}} : (!cir.ptr<!void, target_address_space(1)>, !u32i) -> !void
// CIR: cir.call_llvm_intrinsic "amdgcn.s.prefetch.inst" {{.*}} : (!cir.ptr<!void, target_address_space(4)>, !u32i) -> !void
// LLVM: define{{.*}} void @_Z20test_s_prefetch_instPiPU3AS1fPU3AS4Kcj
// LLVM: call void @llvm.amdgcn.s.prefetch.inst.p0(ptr %{{.*}}, i32 0)
// LLVM: call void @llvm.amdgcn.s.prefetch.inst.p1(ptr addrspace(1) %{{.*}}, i32 %{{.*}})
// LLVM: call void @llvm.amdgcn.s.prefetch.inst.p4(ptr addrspace(4) %{{.*}}, i32 31)
__device__ void test_s_prefetch_inst(int *fp, global_ptr_t gp,
                                     constant_ptr_t cp, unsigned int len) {
  __builtin_amdgcn_s_prefetch_inst(fp, 0);
  __builtin_amdgcn_s_prefetch_inst(gp, len);
  __builtin_amdgcn_s_prefetch_inst(cp, 31);
}
