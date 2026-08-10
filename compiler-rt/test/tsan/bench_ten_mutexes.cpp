// RUN: %clangxx_tsan %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s

// bench.h needs pthread barriers which are not available on OS X
// UNSUPPORTED: darwin

#include "bench.h"

const int kMutex = 10;
pthread_mutex_t mtx[kMutex];

void thread(int tid) {
  for (int i = 0; i < bench_niter; i++) {
    int idx = (i % kMutex);
    if (tid == 0)
      idx = kMutex - idx - 1;
    pthread_mutex_lock(&mtx[idx]);
    pthread_mutex_unlock(&mtx[idx]);
  }
}

void bench() {
  for (int i = 0; i < kMutex; i++)
    pthread_mutex_init(&mtx[i], 0);
  start_thread_group(2, thread);
}

// CHECK: DONE

