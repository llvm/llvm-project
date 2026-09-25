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

// CIR-LABEL: @_Z4ib64y
// CIR: cir.call_llvm_intrinsic "amdgcn.inverse.ballot" %{{.+}} : (!u32i) -> !cir.bool
// LLVM-LABEL: @_Z4ib64y
// LLVM: call i1 @llvm.amdgcn.inverse.ballot.i32(i32
__device__ bool ib64(unsigned long long m) {
  return __builtin_amdgcn_inverse_ballot_w32(m);
}

// CIR-LABEL: @_Z2exv
// CIR: [[TRUE1:%.+]] = cir.const #true
// CIR: cir.call_llvm_intrinsic "amdgcn.ballot" [[TRUE1]] : (!cir.bool) -> !u64i
// LLVM-LABEL: @_Z2exv
// LLVM: call i64 @llvm.amdgcn.ballot.i64(i1 true)
__device__ unsigned long long ex() {
  return __builtin_amdgcn_read_exec();
}

// CIR-LABEL: @_Z4exlov
// CIR: [[TRUE2:%.+]] = cir.const #true
// CIR: cir.call_llvm_intrinsic "amdgcn.ballot" [[TRUE2]] : (!cir.bool) -> !u32i
// LLVM-LABEL: @_Z4exlov
// LLVM: call i32 @llvm.amdgcn.ballot.i32(i1 true)
__device__ unsigned int exlo() {
  return __builtin_amdgcn_read_exec_lo();
}

// CIR-LABEL: @_Z4exhiv
// CIR: [[TRUE3:%.+]] = cir.const #true
// CIR: [[BRES:%.+]] = cir.call_llvm_intrinsic "amdgcn.ballot" [[TRUE3]] : (!cir.bool) -> !u64i
// CIR: [[C32:%.+]] = cir.const #cir.int<32> : !u64i
// CIR: [[SRES:%.+]] = cir.shift(right, [[BRES]] : !u64i, [[C32]] : !u64i) -> !u64i
// CIR: cir.cast integral [[SRES]] : !u64i -> !u32i
// LLVM-LABEL: @_Z4exhiv
// LLVM: [[BRES:%.+]] = call i64 @llvm.amdgcn.ballot.i64(i1 true)
// LLVM: [[SRES:%.+]] = lshr i64 [[BRES]], 32
// LLVM: trunc i64 [[SRES]] to i32
__device__ unsigned int exhi() {
  return __builtin_amdgcn_read_exec_hi();
}
