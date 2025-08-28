// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -g -S -fenable-ripple -O0 -emit-llvm %s
// RUN: %clang -g -S -fenable-ripple -Og -emit-llvm %s
// RUN: %clang -g -S -fenable-ripple -O1 -emit-llvm %s
// RUN: %clang -g -S -fenable-ripple -O2 -emit-llvm %s
// RUN: %clang -g -S -fenable-ripple -O3 -emit-llvm %s
// RUN: %clang -g -S -fenable-ripple -Os -emit-llvm %s
// RUN: %clang -g -S -fenable-ripple -Oz -emit-llvm %s

#include <ripple.h>
#include <stddef.h>

#define S1 8
#define S2 10

void foo(float A[S1][S1], float B[S1][S2]) {
  ripple_block_t BS = ripple_set_block_shape(0, S2, S1);
  size_t thread_x = ripple_id(BS, 0), thread_y = ripple_id(BS, 1);
  float acc = 0.f;
  float pac = 0.f;

  for (unsigned i = 0; i < S1; ++i)
      pac += 1.f;
  if (thread_x < 4) {
    acc = A[thread_y][0];
    for (unsigned i = 0; i < S1; ++i)
      for (unsigned j = 0; j < S2; ++j)
        pac += 1.f;
    if (thread_y < 2)
      for (unsigned i = 0; i < S1; ++i)
        for (unsigned j = 0; j < S2; ++j)
          pac += 1.f;
  }
  for (unsigned j = 0; j < S2; ++j)
    pac += 1.f;

  B[thread_y][thread_x] = acc + pac;
}

#if 0

float A[S1][S1];
float B[S1][S2];

#include <stdlib.h>
#include <stdio.h>
int main(void) {
  printf("A:\n");
  for (unsigned i = 0; i < S1; ++i) {
    for (unsigned j = 0; j < S2; ++j) {
      A[i][j] = i * S2 + j + 1;
      printf("%f ", A[i][j]);
    }
    printf("\n");
  }

  foo(A, B);

  printf("\nB:\n");
  for (unsigned i = 0; i < S1; ++i) {
    for (unsigned j = 0; j < S2; ++j)
      printf("%f ", B[i][j]);
    printf("\n");
  }

  return EXIT_SUCCESS;
}

#endif
