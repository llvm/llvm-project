// RUN: %clang_cc1 -emit-llvm %s -o - -fcuda-is-device -triple nvptx-unknown-unknown | FileCheck %s

#include "../SemaCUDA/cuda.h"

// CHECK: @i = addrspace(1) global
__device__ int i;

// CHECK: @j = addrspace(4) global
__constant__ int j;

// CHECK: @k = addrspace(3) global
__shared__ int k;

__device__ void foo() {
  // CHECK: load i32* bitcast (i32 addrspace(1)* @i to i32*)
  i++;

  // CHECK: load i32* bitcast (i32 addrspace(4)* @j to i32*)
  j++;

  // CHECK: load i32* bitcast (i32 addrspace(3)* @k to i32*)
  k++;

  static int li;
  // CHECK: load i32 addrspace(1)* @_ZZ3foovE2li
  li++;

  __constant__ int lj;
  // CHECK: load i32 addrspace(4)* @_ZZ3foovE2lj
  lj++;

  __shared__ int lk;
  // CHECK: load i32 addrspace(3)* @_ZZ3foovE2lk
  lk++;
}

