#include "../Inputs/cuda.h"

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fclangir \
// RUN:            -fcuda-is-device -emit-cir -target-sdk-version=12.3 \
// RUN:            %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR-DEVICE --input-file=%t.cir %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fclangir \
// RUN:            -fcuda-is-device -emit-llvm -target-sdk-version=12.3 \
// RUN:            %s -o %t.cir
// RUN: FileCheck --check-prefix=LLVM-DEVICE --input-file=%t.cir %s

__device__ int a;
// CIR-DEVICE: cir.global external addrspace(offload_global) @a = #cir.int<0>
// LLVM-DEVICE: @a = addrspace(1) externally_initialized global i32 0, align 4

__shared__ int shared;
// CIR-DEVICE: cir.global external addrspace(offload_local) @shared = #cir.undef
// LLVM-DEVICE: @shared = addrspace(3) global i32 undef, align 4
