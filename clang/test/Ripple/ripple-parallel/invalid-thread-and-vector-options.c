// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -ffreestanding -S -emit-llvm -fenable-ripple %s -o %t.out &>%t.err; FileCheck %s --input-file=%t.err

#include <ripple.h>

#ifndef THREAD_ID
#define THREAD_ID 32
#endif

typedef float f32;

void check(int N, int start, int end, f32 x[__restrict 1][N],
           f32 y[__restrict 1][N], f32 xpy[__restrict 1][N]) {
  ripple_block_t BS = ripple_set_block_shape(THREAD_ID, 32);
  #pragma ripple parallel Block(BS) Dims(0) Thread BlockIndependent
  for (int i = start; i < end; ++i)
    xpy[0][i] = x[0][i] + y[0][i];
}

// CHECK: invalid-thread-and-vector-options.c:15:11: error: unexpected combination of thread and vector arguments to '#pragma ripple parallel ...'
// CHECK-NEXT:    15 |   #pragma ripple parallel Block(BS) Dims(0) Thread BlockIndependent
// CHECK-NEXT:       |           ^
// CHECK: 1 error generated.
