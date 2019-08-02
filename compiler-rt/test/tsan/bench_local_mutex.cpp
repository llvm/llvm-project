// RUN: %clangxx_tsan %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s

// bench.h needs pthread barriers which are not available on OS X
// UNSUPPORTED: darwin

#include "bench.h"

pthread_mutex_t *mtx;
const int kStride = 16;

void thread(int tid) {
  for (int i = 0; i < bench_niter; i++) {
    pthread_mutex_lock(&mtx[tid * kStride]);
    pthread_mutex_unlock(&mtx[tid * kStride]);
  }
}

void bench() {
  mtx = (pthread_mutex_t*)malloc(bench_nthread * kStride * sizeof(*mtx));
  for (int i = 0; i < bench_nthread; i++) {
    pthread_mutex_init(&mtx[i * kStride], 0);
    pthread_mutex_lock(&mtx[i * kStride]);
    pthread_mutex_unlock(&mtx[i * kStride]);
  }
  start_thread_group(bench_nthread, thread);
}

// CHECK: DONE

