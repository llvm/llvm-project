// RUN: %clangxx_tsan %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s

// bench.h needs pthread barriers which are not available on OS X
// UNSUPPORTED: darwin

#include "bench.h"

int *x;
const int kStride = 32;

void thread(int tid) {
  __atomic_load_n(&x[tid * kStride], __ATOMIC_ACQUIRE);
  for (int i = 0; i < bench_niter; i++)
    __atomic_store_n(&x[tid * kStride], 0, __ATOMIC_RELEASE);
}

void bench() {
  x = (int*)malloc(bench_nthread * kStride * sizeof(x[0]));
  for (int i = 0; i < bench_nthread; i++)
    __atomic_store_n(&x[i * kStride], 0, __ATOMIC_RELEASE);
  start_thread_group(bench_nthread, thread);
}

// CHECK: DONE

