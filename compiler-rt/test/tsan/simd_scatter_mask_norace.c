// REQUIRES: mavx512f
// RUN: %clang_tsan -march=native -DSIMDLEN=4 -DTYPE=float %s -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clang_tsan -march=native -DSIMDLEN=4 -DTYPE=double %s -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clang_tsan -march=native -DSIMDLEN=8 -DTYPE=float %s -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clang_tsan -march=native -DSIMDLEN=8 -DTYPE=double %s -o %t && %run %t 2>&1 | FileCheck %s
#include "test.h"
#include <immintrin.h>
#include <stdint.h>

#ifndef SIMDLEN
#  define SIMDLEN 8
#endif /*SIMDLEN*/
#ifndef TYPE
#  define TYPE double
#endif /*TYPE*/

#if SIMDLEN == 4
#  define tsan_scatter_func __tsan_scatter_vector4
#  define intri_type __m256i
#elif SIMDLEN == 8
#  define tsan_scatter_func __tsan_scatter_vector8
#  define intri_type __m512i
#endif

extern void tsan_scatter_func(intri_type, int, uint8_t);
TYPE A[8];

__attribute__((disable_sanitizer_instrumentation)) void *Thread(uint8_t mask) {
#if SIMDLEN == 4
  __m256i vaddr = _mm256_set_epi64x(
#elif SIMDLEN == 8
  __m512i vaddr = _mm512_set_epi64(
      (int64_t)(A + 7), (int64_t)(A + 6), (int64_t)(A + 5), (int64_t)(A + 4),
#endif
      (int64_t)(A + 3), (int64_t)(A + 2), (int64_t)(A + 1), (int64_t)(A + 0));
  tsan_scatter_func(vaddr, sizeof(TYPE), mask);
  barrier_wait(&barrier);
  return NULL;
}

void *Thread1(void *x) { return Thread(0b01010101); }

void *Thread2(void *x) { return Thread(0b10101010); }

int main() {
  barrier_init(&barrier, 2);
  pthread_t t[2];
  pthread_create(&t[0], NULL, Thread1, NULL);
  pthread_create(&t[1], NULL, Thread2, NULL);
  pthread_join(t[0], NULL);
  pthread_join(t[1], NULL);
  fprintf(stderr, "DONE.\n");
  return 0;
}

// CHECK-NOT: WARNING: ThreadSanitizer: data race
// CHECK-NOT: SUMMARY: ThreadSanitizer: data race{{.*}}Thread
