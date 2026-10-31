// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip -fclangir \
// RUN:            -fcuda-is-device -emit-llvm -I%S/Inputs %s -o %t-spirv-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,SPIRV %s --input-file=%t-spirv-cir.ll

// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip \
// RUN:            -fcuda-is-device -emit-llvm -I%S/Inputs %s -o %t-spirv.ll
// RUN: FileCheck --check-prefixes=LLVM,SPIRV %s --input-file=%t-spirv.ll

// RUN: %clang_cc1 -triple amdgpu9-amd-amdhsa -x hip -fclangir \
// RUN:            -fcuda-is-device -emit-llvm -I%S/Inputs %s -o %t-amdgpu-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,AS4 %s --input-file=%t-amdgpu-cir.ll

// RUN: %clang_cc1 -triple amdgpu9-amd-amdhsa -x hip \
// RUN:            -fcuda-is-device -emit-llvm -I%S/Inputs %s -o %t-amdgpu.ll
// RUN: FileCheck --check-prefixes=LLVM,AS4 %s --input-file=%t-amdgpu.ll

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda -fclangir \
// RUN:            -fcuda-is-device -emit-llvm -I%S/Inputs %s -o %t-nvptx-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,AS4 %s --input-file=%t-nvptx-cir.ll

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda \
// RUN:            -fcuda-is-device -emit-llvm -I%S/Inputs %s -o %t-nvptx.ll
// RUN: FileCheck --check-prefixes=LLVM,AS4 %s --input-file=%t-nvptx.ll

#include "cuda.h"

struct S {
  int x;
  __device__ S() {}
  __device__ ~S() {}
};

__device__ S d;
__constant__ S c;

// LLVM: @d = addrspace(1) externally_initialized global %struct.S zeroinitializer, align 4
// SPIRV: @c = addrspace(1) externally_initialized constant %struct.S zeroinitializer, align 4
// AS4: @c = addrspace(4) externally_initialized constant %struct.S zeroinitializer, align 4
// LLVM-NOT: llvm.global_ctors
// LLVM-NOT: llvm.global_dtors
