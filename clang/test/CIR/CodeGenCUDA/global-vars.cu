#include "Inputs/cuda.h"

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fclangir \
// RUN:            -fcuda-is-device -emit-cir -target-sdk-version=12.3 \
// RUN:            -I%S/Inputs/ %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR-DEVICE --input-file=%t.cir %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:            -x cuda -emit-cir -target-sdk-version=12.3 \
// RUN:            -I%S/Inputs/ %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR-HOST --input-file=%t.cir %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fclangir \
// RUN:            -fcuda-is-device -emit-llvm -target-sdk-version=12.3 \
// RUN:            -I%S/Inputs/ %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM-DEVICE --input-file=%t.ll %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:            -x cuda -emit-llvm -target-sdk-version=12.3 \
// RUN:            -I%S/Inputs/ %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM-HOST --input-file=%t.ll %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda \
// RUN:            -fcuda-is-device -emit-llvm -target-sdk-version=12.3 \
// RUN:            -I%S/Inputs/ %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG-DEVICE --input-file=%t.ll %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu \
// RUN:            -x cuda -emit-llvm -target-sdk-version=12.3 \
// RUN:            -I%S/Inputs/ %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG-HOST --input-file=%t.ll %s

__shared__ int a;
// CIR-DEVICE: cir.global external [[SHARED:@.*]] = #cir.int<0> : !s32i {alignment = 4 : i64}
// CIR-HOST: cir.global external [[SHARED_HOST:@.*]] = #cir.int<0> : !s32i {alignment = 4 : i64}
// LLVM-DEVICE: @[[SHARED_LL:.*]] = global i32 0, align 4
// LLVM-HOST: @[[SHARED_LH:.*]] = global i32 0, align 4
// OGCG-DEVICE: @[[SHARED_OD:.*]] = addrspace(3) global i32 undef, align 4
// OGCG-HOST: @[[SHARED_OH:.*]] = internal global i32 undef, align 4

__device__ int b;
// CIR-DEVICE: cir.global external [[DEV:@.*]] = #cir.int<0> : !s32i {alignment = 4 : i64}
// CIR-HOST: cir.global external [[DEV_HOST:@.*]] = #cir.int<0> : !s32i {alignment = 4 : i64}
// LLVM-DEVICE: @[[DEV_LD:.*]] = global i32 0, align 4
// LLVM-HOST: @[[DEV_LH:.*]] = global i32 0, align 4
// OGCG-HOST: @[[DEV_OH:.*]] = internal global i32 undef, align 4
// OGCG-DEVICE: @[[DEV_OD:.*]] = addrspace(1) externally_initialized global i32 0, align 4
