#include "../Inputs/cuda.h"

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fclangir \
// RUN:            -fcuda-is-device -emit-cir -target-sdk-version=12.3 \
// RUN:            %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR-DEVICE --input-file=%t.cir %s


__device__ int a;

// CIR-DEVICE: cir.global external addrspace(offload_global) @a = #cir.int<0> : !s32i {alignment = 4 : i64} loc(#loc3)