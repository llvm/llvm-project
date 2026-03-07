#include "Inputs/cuda.h"

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fclangir \
// RUN:            -fcuda-is-device -emit-cir -target-sdk-version=12.3 \
// RUN:            -I%S/Inputs/ %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR-DEVICE --input-file=%t.cir %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fclangir \
// RUN:            -fcuda-is-device -emit-llvm -target-sdk-version=12.3 \
// RUN:            -I%S/Inputs/ %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM-DEVICE --input-file=%t.ll %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda \
// RUN:            -fcuda-is-device -emit-llvm -target-sdk-version=12.3 \
// RUN:            -I%S/Inputs/ %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG-DEVICE --input-file=%t.ll %s


__device__ int a;
// CIR-DEVICE: cir.global external @[[DEV:.*]] = #cir.int<0> : !s32i {alignment = 4 : i64, cu.externally_initialized = #cir.cu.externally_initialized}
// LLVM-DEVICE: @[[DEV_LD:.*]] = externally_initialized global i32 0, align 4
// OGCG-DEVICE: @[[DEV_OD:.*]] = addrspace(1) externally_initialized global i32 0, align 4

__shared__ int b;
// CIR-DEVICE: cir.global external @[[SHARED:.*]] = #cir.undef : !s32i {alignment = 4 : i64}
// LLVM-DEVICE: @[[SHARED_LL:.*]] = global i32 undef, align 4
// OGCG-DEVICE: @[[SHARED_OD:.*]] = addrspace(3) global i32 undef, align 4

__constant__ int c;
// CIR-DEVICE: cir.global constant external @[[CONST:.*]] = #cir.int<0> : !s32i {alignment = 4 : i64, cu.externally_initialized = #cir.cu.externally_initialized}
// LLVM-DEVICE: @[[CONST_LL:.*]] = externally_initialized constant i32 0, align 4
// OGCG-DEVICE: @[[CONST_OD:.*]] = addrspace(4) externally_initialized constant i32 0, align 4
