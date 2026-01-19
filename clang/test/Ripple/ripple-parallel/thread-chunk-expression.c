// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -ffreestanding -S -emit-llvm -fenable-ripple %s -o %t.out &>%t.err; FileCheck %s --input-file=%t.err

#include <ripple.h>

#ifndef THREAD_ID
#define THREAD_ID 32
#endif

typedef float f32;

void check(int Chunk, int N, int start, int end, f32 x[restrict N],
           f32 y[restrict N], f32 xpy[restrict N]) {
  ripple_block_t BS = ripple_set_block_shape(THREAD_ID, 32);
  #pragma ripple parallel Block(BS) Dims(0) ThreadChunk(3 + chunk)
  for (int i = start; i < end; ++i)
    xpy[i] = x[i] + y[i];
}

void check2(int Chunk, int N, int start, int end, f32 x[restrict N],
           f32 y[restrict N], f32 xpy[restrict N]) {
  ripple_block_t BS = ripple_set_block_shape(THREAD_ID, 32);
  #pragma ripple parallel Block(BS) Dims(0) ThreadChunk(Chunk + 3)
  for (int i = start; i < end; ++i)
    xpy[i] = x[i] + y[i];
}

// CHECK:      thread-chunk-expression.c:15:58: error: expected ')' after '3'
// CHECK-NEXT:    15 |   #pragma ripple parallel Block(BS) Dims(0) ThreadChunk(3 + chunk)
// CHECK-NEXT:       |                                                          ^
// CHECK-NEXT:       |                                                          )
// CHECK-NEXT: thread-chunk-expression.c:23:62: error: expected ')' after 'Chunk'
// CHECK-NEXT:    23 |   #pragma ripple parallel Block(BS) Dims(0) ThreadChunk(Chunk + 3)
// CHECK-NEXT:       |                                                              ^
// CHECK-NEXT:       |                                                              )
// CHECK-NEXT: 2 errors generated.
