// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -cl-std=CL2.0 -fclangir -emit-cir %s -o %t-amdgcn.cir
// RUN: FileCheck --input-file=%t-amdgcn.cir %s -check-prefix=CIR-AMDGCN
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -cl-std=CL2.0 -fclangir -emit-llvm %s -o %t-amdgcn-cir.ll
// RUN: FileCheck --input-file=%t-amdgcn-cir.ll %s -check-prefix=LLVM-AMDGCN
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -cl-std=CL2.0 -emit-llvm %s -o %t-amdgcn.ll
// RUN: FileCheck --input-file=%t-amdgcn.ll %s -check-prefix=LLVM-AMDGCN
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -cl-std=CL2.0 -fclangir -emit-cir %s -o %t-nvptx.cir
// RUN: FileCheck --input-file=%t-nvptx.cir %s -check-prefix=CIR-NVPTX
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -cl-std=CL2.0 -fclangir -emit-llvm %s -o %t-nvptx-cir.ll
// RUN: FileCheck --input-file=%t-nvptx-cir.ll %s -check-prefix=LLVM-NVPTX
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -cl-std=CL2.0 -emit-llvm %s -o %t-nvptx.ll
// RUN: FileCheck --input-file=%t-nvptx.ll %s -check-prefix=LLVM-NVPTX

// OpenCL kernels use the target's device kernel calling convention.

kernel void k(global int *p) { *p = 1; }

// CIR-AMDGCN: cir.func {{.*}}@k({{.*}}cc(amdgpu_kernel)
// LLVM-AMDGCN: define dso_local amdgpu_kernel void @k(

// CIR-NVPTX: cir.func {{.*}}@k({{.*}}cc(ptx_kernel)
// LLVM-NVPTX: define dso_local ptx_kernel void @k(
