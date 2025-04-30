// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple=x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s

#include <next32_scratchpad.h>
#include <stdint.h>

__next32_tls__ __thread int32_t arr1[34];
__next32_global__ int32_t arr2[34];
__next32_constant__ int32_t arr3[34];
// In the standard C11 header for threading <threads.h>, keyword thread_local is
// an alias of _Thread_local. Keyword thread_local is a part of C++11 standard.
__next32_tls__ _Thread_local int32_t arr4[34];

__next32_local__ int32_t arr5[34];

int foo() {
  return arr1[0] + arr2[1] + arr3[3] + arr4[4] + arr5[5];
  // CHECK: load i32, ptr addrspace(273) %arrayidx, align 16
  // CHECK: load i32, ptr addrspace(274) getelementptr inbounds ([34 x i32], ptr addrspace(274) @arr2, i64 0, i64 1), align 4
  // CHECK: load i32, ptr addrspace(275) getelementptr inbounds ([34 x i32], ptr addrspace(275) @arr3, i64 0, i64 3), align 4
  // CHECK: load i32, ptr addrspace(273) %arrayidx2, align 16
  // CHECK: load i32, ptr addrspace(3) getelementptr inbounds ([34 x i32], ptr addrspace(3) @arr5, i64 0, i64 5), align 4
}
