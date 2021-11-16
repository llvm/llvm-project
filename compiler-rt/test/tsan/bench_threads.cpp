// RUN: %clangxx_tsan %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s

// bench.h needs pthread barriers which are not available on OS X
// UNSUPPORTED: darwin

#include "bench.h"

void *nop_thread(void *arg) {
  pthread_setname_np(pthread_self(), "nop_thread");
  return nullptr;
}

void thread(int tid) {
  for (int i = 0; i < bench_niter; i++) {
    pthread_t th;
    pthread_create(&th, nullptr, nop_thread, nullptr);
    pthread_join(th, nullptr);
  }
}

void bench() {
  // Benchmark thread creation/joining in presence of a large number
  // of threads (both alive and already joined).
  printf("starting transient threads...\n");
  for (int i = 0; i < 200; i++) {
    const int kBatch = 100;
    pthread_t th[kBatch];
    for (int j = 0; j < kBatch; j++)
      pthread_create(&th[j], nullptr, nop_thread, nullptr);
    for (int j = 0; j < kBatch; j++)
      pthread_join(th[j], nullptr);
  }
  printf("starting persistent threads...\n");
  const int kLiveThreads = 2000;
  pthread_t th[kLiveThreads];
  for (int j = 0; j < kLiveThreads; j++)
    pthread_create(&th[j], nullptr, nop_thread, nullptr);
  printf("starting benchmark threads...\n");
  start_thread_group(bench_nthread, thread);
  for (int j = 0; j < kLiveThreads; j++)
    pthread_join(th[j], nullptr);
}

// CHECK: DONE
