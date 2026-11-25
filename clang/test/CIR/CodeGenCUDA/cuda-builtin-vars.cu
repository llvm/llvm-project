// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda \
// RUN:   -fcuda-is-device -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda \
// RUN:   -fcuda-is-device -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda \
// RUN:   -fcuda-is-device -emit-llvm %s -o %t.ogcg.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ogcg.ll %s

#include "__clang_cuda_builtin_vars.h"

__attribute__((global))
void kernel(int *out) {
  int i = 0;
  out[i++] = threadIdx.x;
  out[i++] = threadIdx.y;
  out[i++] = threadIdx.z;

  out[i++] = blockIdx.x;
  out[i++] = blockIdx.y;
  out[i++] = blockIdx.z;

  out[i++] = blockDim.x;
  out[i++] = blockDim.y;
  out[i++] = blockDim.z;

  out[i++] = gridDim.x;
  out[i++] = gridDim.y;
  out[i++] = gridDim.z;
}

// CIR-DAG: cir.call_llvm_intrinsic "nvvm.read.ptx.sreg.tid.x"
// CIR-DAG: cir.call_llvm_intrinsic "nvvm.read.ptx.sreg.tid.y"
// CIR-DAG: cir.call_llvm_intrinsic "nvvm.read.ptx.sreg.tid.z"
// CIR-DAG: cir.call_llvm_intrinsic "nvvm.read.ptx.sreg.ctaid.x"
// CIR-DAG: cir.call_llvm_intrinsic "nvvm.read.ptx.sreg.ctaid.y"
// CIR-DAG: cir.call_llvm_intrinsic "nvvm.read.ptx.sreg.ctaid.z"
// CIR-DAG: cir.call_llvm_intrinsic "nvvm.read.ptx.sreg.ntid.x"
// CIR-DAG: cir.call_llvm_intrinsic "nvvm.read.ptx.sreg.ntid.y"
// CIR-DAG: cir.call_llvm_intrinsic "nvvm.read.ptx.sreg.ntid.z"
// CIR-DAG: cir.call_llvm_intrinsic "nvvm.read.ptx.sreg.nctaid.x"
// CIR-DAG: cir.call_llvm_intrinsic "nvvm.read.ptx.sreg.nctaid.y"
// CIR-DAG: cir.call_llvm_intrinsic "nvvm.read.ptx.sreg.nctaid.z"

// LLVM-DAG: call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
// LLVM-DAG: call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
// LLVM-DAG: call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.z()
// LLVM-DAG: call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
// LLVM-DAG: call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
// LLVM-DAG: call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.z()
// LLVM-DAG: call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
// LLVM-DAG: call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
// LLVM-DAG: call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.z()
// LLVM-DAG: call noundef i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
// LLVM-DAG: call noundef i32 @llvm.nvvm.read.ptx.sreg.nctaid.y()
// LLVM-DAG: call noundef i32 @llvm.nvvm.read.ptx.sreg.nctaid.z()

