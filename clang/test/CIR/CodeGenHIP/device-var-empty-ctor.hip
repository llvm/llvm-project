// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip -fclangir \
// RUN:            -fcuda-is-device -emit-llvm -I%S/../CodeGenCUDA/Inputs %s -o %t-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,SPIRV %s --input-file=%t-cir.ll

// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip \
// RUN:            -fcuda-is-device -emit-llvm -I%S/../CodeGenCUDA/Inputs %s -o %t.ll
// RUN: FileCheck --check-prefixes=LLVM,SPIRV %s --input-file=%t.ll

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fclangir \
// RUN:            -fcuda-is-device -emit-llvm -I%S/../CodeGenCUDA/Inputs %s -o %t-amdgcn-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,AMDGCN %s --input-file=%t-amdgcn-cir.ll

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip \
// RUN:            -fcuda-is-device -emit-llvm -I%S/../CodeGenCUDA/Inputs %s -o %t-amdgcn.ll
// RUN: FileCheck --check-prefixes=LLVM,AMDGCN %s --input-file=%t-amdgcn.ll

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
// AMDGCN: @c = addrspace(4) externally_initialized constant %struct.S zeroinitializer, align 4
// LLVM-NOT: llvm.global_ctors
// LLVM-NOT: llvm.global_dtors
