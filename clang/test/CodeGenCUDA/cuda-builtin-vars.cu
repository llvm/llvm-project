// RUN: %clang_cc1 "-triple" "nvptx-nvidia-cuda" -emit-llvm -fcuda-is-device -o - %s | FileCheck %s

#include "__clang_cuda_builtin_vars.h"

// CHECK: define{{.*}} void @_Z6kernelPi(ptr noundef %out)
__attribute__((global))
void kernel(int *out) {
  int i = 0;
  out[i++] = threadIdx.x; // CHECK: call noundef{{.*}} i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  out[i++] = threadIdx.y; // CHECK: call noundef{{.*}} i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  out[i++] = threadIdx.z; // CHECK: call noundef{{.*}} i32 @llvm.nvvm.read.ptx.sreg.tid.z()

  out[i++] = blockIdx.x; // CHECK: call noundef{{.*}} i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  out[i++] = blockIdx.y; // CHECK: call noundef{{.*}} i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  out[i++] = blockIdx.z; // CHECK: call noundef{{.*}} i32 @llvm.nvvm.read.ptx.sreg.ctaid.z()

  out[i++] = blockDim.x; // CHECK: call noundef{{.*}} i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  out[i++] = blockDim.y; // CHECK: call noundef{{.*}} i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
  out[i++] = blockDim.z; // CHECK: call noundef{{.*}} i32 @llvm.nvvm.read.ptx.sreg.ntid.z()

  out[i++] = gridDim.x; // CHECK: call noundef{{.*}} i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
  out[i++] = gridDim.y; // CHECK: call noundef{{.*}} i32 @llvm.nvvm.read.ptx.sreg.nctaid.y()
  out[i++] = gridDim.z; // CHECK: call noundef{{.*}} i32 @llvm.nvvm.read.ptx.sreg.nctaid.z()

  // CHECK: [[TID:%.*]] = call noundef{{.*}} i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  // CHECK: [[TID_CMP:%.*]] = icmp ult i32 [[TID]], 1024
  // CHECK: call void @llvm.assume(i1 [[TID_CMP]])
  __builtin_assume(threadIdx.x < 1024);

  // CHECK: [[CTAID:%.*]] = call noundef{{.*}} i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  // CHECK: [[CTAID_CMP:%.*]] = icmp ult i32 [[CTAID]], 1024
  // CHECK: call void @llvm.assume(i1 [[CTAID_CMP]])
  __builtin_assume(blockIdx.x < 1024);

  // CHECK: [[NTID:%.*]] = call noundef{{.*}} i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  // CHECK: [[NTID_CMP:%.*]] = icmp ult i32 [[NTID]], 1024
  // CHECK: call void @llvm.assume(i1 [[NTID_CMP]])
  __builtin_assume(blockDim.x < 1024);

  // CHECK: [[NCTAID:%.*]] = call noundef{{.*}} i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
  // CHECK: [[NCTAID_CMP:%.*]] = icmp ult i32 [[NCTAID]], 1024
  // CHECK: call void @llvm.assume(i1 [[NCTAID_CMP]])
  __builtin_assume(gridDim.x < 1024);

  out[i++] = warpSize; // CHECK: store i32 32,

  // CHECK: ret void
}
