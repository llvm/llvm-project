#include "Inputs/cuda.h"

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fclangir \
// RUN:            -fcuda-is-device -emit-cir -target-sdk-version=12.3 \
// RUN:            -fvisibility=hidden -fapply-global-visibility-to-externs \
// RUN:            -I%S/Inputs/ %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR-DEVICE --input-file=%t.cir %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fclangir \
// RUN:            -fcuda-is-device -emit-llvm -target-sdk-version=12.3 \
// RUN:            -fvisibility=hidden -fapply-global-visibility-to-externs \
// RUN:            -I%S/Inputs/ %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM-DEVICE --input-file=%t.ll %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda \
// RUN:            -fcuda-is-device -emit-llvm -target-sdk-version=12.3 \
// RUN:            -fvisibility=hidden -fapply-global-visibility-to-externs \
// RUN:            -I%S/Inputs/ %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG-DEVICE --input-file=%t.ll %s

// CIR-DEVICE: cir.global protected {{.*}} @deviceVar = #cir.int<0>
// LLVM-DEVICE: @deviceVar = protected addrspace(1) externally_initialized global i32 0
// OGCG-DEVICE: @deviceVar = protected addrspace(1) externally_initialized global i32 0
__attribute__((device)) __device__ int deviceVar;

// CIR-DEVICE: cir.global protected constant {{.*}} @constantVar = #cir.int<0>
// LLVM-DEVICE: @constantVar = protected addrspace(4) externally_initialized constant i32 0
// OGCG-DEVICE: @constantVar = protected addrspace(4) externally_initialized constant i32 0
__attribute__((constant)) __constant__ int constantVar;

// CIR-DEVICE: cir.global protected {{.*}} @nonconstVal = #cir.int<42>
// LLVM-DEVICE: @nonconstVal = protected addrspace(1) externally_initialized global i32 42
// OGCG-DEVICE: @nonconstVal = protected addrspace(1) externally_initialized global i32 42
__device__ int nonconstVal = 42;

// CIR-DEVICE: cir.func {{.*}} protected {{.*}} @_Z10kernelFuncv()
// LLVM-DEVICE: define protected ptx_kernel void @_Z10kernelFuncv()
// OGCG-DEVICE: define protected ptx_kernel void @_Z10kernelFuncv()
__attribute__((global)) __global__ void kernelFunc() {
}
