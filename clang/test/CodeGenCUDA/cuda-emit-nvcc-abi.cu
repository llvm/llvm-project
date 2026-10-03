// RUN: echo "" > %t.fatbin
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-sdk-version=12.0 \
// RUN:   -fgpu-rdc --cuda-emit-nvcc-abi -emit-llvm -fcuda-include-gpubinary \
// RUN:   %t.fatbin -o - -x cuda %s \
// RUN:   | FileCheck --check-prefix=NVCC %s

#include "Inputs/cuda.h"

__global__ void kernel() {}
__device__ int var = 0;

// NVCC: @[[FATBIN:[0-9]+]] = private constant {{.*}}, section "__nv_relfatbin"
// NVCC: @__cuda_fatbin_wrapper = internal constant { i32, i32, ptr, ptr } { i32 1180844977, i32 1, ptr @[[FATBIN]], ptr null }, section ".nvFatBinSegment"
// NVCC: @__fatbinwrap__nv_[[ID:[0-9a-f]+]] = alias { i32, i32, ptr, ptr }, ptr @__cuda_fatbin_wrapper
// NVCC: define internal void @__cuda_register_globals(ptr %{{.*}})
// NVCC: define internal void @__cuda_module_ctor()
// NVCC: call void @__cudaRegisterLinkedBinary__nv_[[ID]](ptr @__cuda_register_globals, ptr @__cuda_fatbin_wrapper, ptr @{{.*}}, ptr @dummy)
// NVCC-NOT: @.offloading.entry
// NVCC-NOT: __tgt_offload_entry
