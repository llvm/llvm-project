// RUN: %clang_tsan -DSIMDLEN=4 -DTYPE=float -O3 %avx2 -fopenmp-simd %s -o %t && %deflake %run %t 2>&1 | FileCheck %s
// RUN: %clang_tsan -DSIMDLEN=4 -DTYPE=double -O3 %avx2 -fopenmp-simd %s -o %t && %deflake %run %t 2>&1 | FileCheck %s
// RUN: %clang_tsan -DSIMDLEN=8 -DTYPE=float -O3 %avx512f -fopenmp-simd %s -o %t && %deflake %run %t 2>&1 | FileCheck %s
// RUN: %clang_tsan -DSIMDLEN=8 -DTYPE=double -O3 %avx512f -fopenmp-simd %s -o %t && %deflake %run %t 2>&1 | FileCheck %s
#include "test.h"

#ifndef SIMDLEN
#  define SIMDLEN 8
#endif /*SIMDLEN*/
#ifndef TYPE
#  define TYPE double
#endif /*TYPE*/
#define LEN 256
#define CHUNK_SIZE 64

TYPE A[2 * LEN];
TYPE B[LEN];

void *Thread(intptr_t offset) {
  for (intptr_t i = offset; i < LEN; i += (2 * CHUNK_SIZE)) {
#pragma omp simd simdlen(SIMDLEN)
    for (intptr_t j = i; j < i + CHUNK_SIZE; j++)
      A[j + 64] = A[j] + B[j];
  }
  barrier_wait(&barrier);
  return NULL;
}

void *Thread1(void *x) { return Thread(0); }

void *Thread2(void *x) { return Thread(CHUNK_SIZE); }

int main() {
  barrier_init(&barrier, 2);
  pthread_t t[2];
  pthread_create(&t[0], NULL, Thread1, NULL);
  pthread_create(&t[1], NULL, Thread2, NULL);
  pthread_join(t[0], NULL);
  pthread_join(t[1], NULL);
  return 0;
}

// CHECK: WARNING: ThreadSanitizer: data race
// CHECK: SUMMARY: ThreadSanitizer: data race{{.*}}Thread
