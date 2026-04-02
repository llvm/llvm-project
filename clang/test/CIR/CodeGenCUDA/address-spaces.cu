#include "Inputs/cuda.h"

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fclangir \
// RUN:            -fcuda-is-device -emit-cir -target-sdk-version=12.3 \
// RUN:            -I%S/Inputs/ %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR-DEVICE --input-file=%t.cir %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda \
// RUN:   -fcuda-is-device -fclangir -emit-cir \
// RUN:   -mmlir -mlir-print-ir-before=cir-target-lowering %s -o %t.cir 2> %t-pre.cir
// RUN: FileCheck --check-prefix=CIR-PRE --input-file=%t-pre.cir %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda \
// RUN:   -fcuda-is-device -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR-POST --input-file=%t.cir %s

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

// CIR-PRE: cir.global external  lang_address_space(offload_global) @i = #cir.int<0> : !s32i
// CIR-POST: cir.global external  target_address_space(1) @i = #cir.int<0> : !s32i
// LLVM-DEVICE-DAG: @i = addrspace(1) {{.*}}global i32 0, align 4
// OGCG-DAG: @i = addrspace(1) externally_initialized global i32 0, align 4
__device__ int i;

// CIR-PRE: cir.global constant external  lang_address_space(offload_constant) @j = #cir.int<0> : !s32i
// CIR-POST: cir.global constant external  target_address_space(4) @j = #cir.int<0> : !s32i
// LLVM-DEVICE-DAG: @j = addrspace(4) {{.*}}constant i32 0, align 4
// OGCG-DAG: @j = addrspace(4) externally_initialized constant i32 0, align 4
__constant__ int j;

// CIR-PRE: cir.global external  lang_address_space(offload_local) @k = #cir.poison : !s32i
// CIR-POST: cir.global external  target_address_space(3) @k = #cir.poison : !s32i
// LLVM-DEVICE-DAG: @k = addrspace(3) global i32 {{undef|poison}}, align 4
// OGCG-DAG: @k = addrspace(3) global i32 undef, align 4
__shared__ int k;

// CIR-PRE: cir.global external  lang_address_space(offload_local) @b = #cir.poison : !cir.float
// CIR-POST: cir.global external  target_address_space(3) @b = #cir.poison : !cir.float
// LLVM-DEVICE-DAG: @b = addrspace(3) global float {{undef|poison}}, align 4
// OGCG-DAG: @b = addrspace(3) global float undef, align 4
__shared__ float b;

__device__ void foo() {
  // CIR-PRE: cir.get_global @i : !cir.ptr<!s32i, lang_address_space(offload_global)>
  // CIR-POST: cir.get_global @i : !cir.ptr<!s32i, target_address_space(1)>
  i++;

  // CIR-PRE: cir.get_global @j : !cir.ptr<!s32i, lang_address_space(offload_constant)>
  // CIR-POST: cir.get_global @j : !cir.ptr<!s32i, target_address_space(4)>
  j++;

  // CIR-PRE: cir.get_global @k : !cir.ptr<!s32i, lang_address_space(offload_local)>
  // CIR-POST: cir.get_global @k : !cir.ptr<!s32i, target_address_space(3)>
  k++;
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
