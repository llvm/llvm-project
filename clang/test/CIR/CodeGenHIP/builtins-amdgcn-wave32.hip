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

#define __device__ __attribute__((device))

//===----------------------------------------------------------------------===//
// Test AMDGPU builtins
//===----------------------------------------------------------------------===//

// CIR-LABEL: @_Z15test_ballot_w32Pjii
// CIR: [[CMP:%.+]] = cir.cmp eq %{{.+}}, %{{.+}} : !s32i
// CIR: cir.call_llvm_intrinsic "amdgcn.ballot" [[CMP]] : (!cir.bool) -> !u32i
// LLVM: define{{.*}} void @_Z15test_ballot_w32Pjii
// LLVM: [[CMP:%.+]] = icmp eq i32 %{{.+}}, %{{.+}}
// LLVM: call{{.*}} i32 @llvm.amdgcn.ballot.i32(i1 [[CMP]])
__device__ void test_ballot_w32(unsigned int* out, int a, int b) {
  *out = __builtin_amdgcn_ballot_w32(a == b);
}
