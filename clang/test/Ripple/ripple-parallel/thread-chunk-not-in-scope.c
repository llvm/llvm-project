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
  #pragma ripple parallel Block(BS) Dims(0) ThreadChunk(chunk)
  for (int i = start; i < end; ++i)
    xpy[i] = x[i] + y[i];
}

// CHECK: thread-chunk-not-in-scope.c:15:57: error: use of undeclared identifier 'chunk'; did you mean 'Chunk'?
// CHECK:    15 |   #pragma ripple parallel Block(BS) Dims(0) ThreadChunk(chunk)
// CHECK:       |                                                         ^~~~~
// CHECK:       |                                                         Chunk
// CHECK: 1 error generated.
