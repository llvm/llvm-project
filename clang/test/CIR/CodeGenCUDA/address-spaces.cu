#include "Inputs/cuda.h"

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fclangir \
// RUN:            -fcuda-is-device -emit-cir -target-sdk-version=12.3 \
// RUN:            -I%S/Inputs/ %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR-DEVICE --input-file=%t.cir %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda \
// RUN:   -fcuda-is-device -fclangir -emit-cir \
// RUN:   -mmlir -mlir-print-ir-before=cir-target-lowering %s -o %t.cir 2> %t-pre.cir
// RUN: FileCheck --check-prefix=CIR-PRE --input-file=%t-pre.cir %s

// TODO: Add CIR (post target lowering) and LLVM checks once NVPTX TargetLoweringInfo
// is implemented.

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fclangir \
// RUN:            -fcuda-is-device -emit-llvm -target-sdk-version=12.3 \
// RUN:            -I%S/Inputs/ %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM-DEVICE --input-file=%t.ll %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda \
// RUN:   -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda \
// RUN:            -fcuda-is-device -emit-llvm -target-sdk-version=12.3 \
// RUN:            -I%S/Inputs/ %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG-DEVICE --input-file=%t.ll %s

// Verifies CIR emits correct address spaces for CUDA globals.

// CIR-DEVICE: cir.global "private" internal dso_local @_ZZ2fnvE1j = #cir.undef : !s32i {alignment = 4 : i64}
// LLVM-DEVICE: @_ZZ2fnvE1j = internal global i32 undef, align 4

__device__ int a;
// CIR-PRE: cir.global external lang_address_space(offload_global) @a = #cir.int<0> : !s32i {alignment = 4 : i64, cu.externally_initialized = #cir.cu.externally_initialized}
// LLVM-DEVICE: @[[DEV_LD:.*]] = externally_initialized global i32 0, align 4
// OGCG-DAG: @a = addrspace(1) externally_initialized global i32 0, align 4
// OGCG-DEVICE: @[[DEV_OD:.*]] = addrspace(1) externally_initialized global i32 0, align 4

__constant__ int c;
// CIR-PRE: cir.global constant external lang_address_space(offload_constant) @c = #cir.int<0> : !s32i {alignment = 4 : i64, cu.externally_initialized = #cir.cu.externally_initialized}
// LLVM-DEVICE: @[[CONST_LL:.*]] = externally_initialized constant i32 0, align 4
// OGCG-DAG: @c = addrspace(4) externally_initialized constant i32 0, align 4
// OGCG-DEVICE: @[[CONST_OD:.*]] = addrspace(4) externally_initialized constant i32 0, align 4

// OGCG-DEVICE: @_ZZ2fnvE1j = internal addrspace(3) global i32 undef, align 4

__device__ void foo() {
  // CIR-PRE: cir.get_global @a : !cir.ptr<!s32i, lang_address_space(offload_global)>
  a++;

  // CIR-PRE: cir.get_global @c : !cir.ptr<!s32i, lang_address_space(offload_constant)>
  c++;
}

__global__ void fn() {
  int i = 0;
  __shared__ int j;
  j = i;
}

// CIR-DEVICE: cir.func {{.*}}@_Z2fnv() {{.*}} {
// CIR-DEVICE:   %[[ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init]
// CIR-DEVICE:   %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR-DEVICE:   cir.store {{.*}}%[[ZERO]], %[[ALLOCA]] : !s32i, !cir.ptr<!s32i>
// CIR-DEVICE:   %[[J:.*]] = cir.get_global @_ZZ2fnvE1j : !cir.ptr<!s32i>
// CIR-DEVICE:   %[[VAL:.*]] = cir.load {{.*}}%[[ALLOCA]] : !cir.ptr<!s32i>, !s32i
// CIR-DEVICE:   cir.store {{.*}}%[[VAL]], %[[J]] : !s32i, !cir.ptr<!s32i>
// CIR-DEVICE:   cir.return

// LLVM-DEVICE: define dso_local void @_Z2fnv()
// LLVM-DEVICE:   %[[ALLOCA:.*]] = alloca i32, i64 1, align 4
// LLVM-DEVICE:   store i32 0, ptr %[[ALLOCA]], align 4
// LLVM-DEVICE:   %[[VAL:.*]] = load i32, ptr %[[ALLOCA]], align 4
// LLVM-DEVICE:   store i32 %[[VAL]], ptr @_ZZ2fnvE1j, align 4
// LLVM-DEVICE:   ret void

// OGCG-DEVICE: define dso_local ptx_kernel void @_Z2fnv()
// OGCG-DEVICE: entry:
// OGCG-DEVICE:   %[[ALLOCA:.*]] = alloca i32, align 4
// OGCG-DEVICE:   store i32 0, ptr %[[ALLOCA]], align 4
// OGCG-DEVICE:   %[[VAL:.*]] = load i32, ptr %[[ALLOCA]], align 4
// OGCG-DEVICE:   store i32 %[[VAL]], ptr addrspacecast (ptr addrspace(3) @_ZZ2fnvE1j to ptr), align 4
// OGCG-DEVICE:   ret void
