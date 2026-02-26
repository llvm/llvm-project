// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda \
// RUN:   -fcuda-is-device -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// Verifies CIR emits correct address spaces for CUDA globals.

#include "Inputs/cuda.h"

// CHECK: cir.global external  lang_address_space(offload_global) @i = #cir.int<0> : !s32i
__device__ int i;

// CHECK: cir.global constant external  lang_address_space(offload_constant) @j = #cir.int<0> : !s32i
__constant__ int j;

// CHECK: cir.global external  lang_address_space(offload_local) @k = #cir.poison : !s32i
__shared__ int k;

// CHECK: cir.global external  lang_address_space(offload_local) @b = #cir.poison : !cir.float
__shared__ float b;

__device__ void foo() {
  // CHECK: cir.get_global @i : !cir.ptr<!s32i, lang_address_space(offload_global)>
  i++;

  // CHECK: cir.get_global @j : !cir.ptr<!s32i, lang_address_space(offload_constant)>
  j++;

  // CHECK: cir.get_global @k : !cir.ptr<!s32i, lang_address_space(offload_local)>
  k++;
}
