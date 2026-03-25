// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -ffreestanding -S -emit-llvm -fenable-ripple %s -o %t.out &>%t.err; FileCheck %s --input-file=%t.err

#include "../ripple_test.h"

#ifndef THREAD_ID
#define THREAD_ID 32
#endif

typedef float f32;

void check(int Chunk, int N, int start, int end, f32 x[restrict N],
           f32 y[restrict N], f32 xpy[restrict N]) {
  ripple_thd_block_t ThreadBlock = ripple_thd_init(0, NULL);
  #pragma ripple parallel Block(ThreadBlock) Dims(0) ThreadChunk(chunk)
  for (int i = start; i < end; ++i)
    xpy[i] = x[i] + y[i];
  ripple_thd_exit(ThreadBlock);
}

// CHECK: thread-chunk-not-in-scope.c:15:66: error: use of undeclared identifier 'chunk'; did you mean 'Chunk'?
// CHECK:    15 |   #pragma ripple parallel Block(ThreadBlock) Dims(0) ThreadChunk(chunk)
// CHECK:       |                                                                  ^~~~~
// CHECK:       |                                                                  Chunk
// CHECK: 1 error generated.
