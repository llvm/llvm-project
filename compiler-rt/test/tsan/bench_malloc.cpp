// RUN: %clangxx_tsan %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s

// bench.h needs pthread barriers which are not available on OS X
// UNSUPPORTED: darwin

#include "bench.h"

void thread(int tid) {
  void **blocks = new void *[bench_mode];
  for (int i = 0; i < bench_niter; i++) {
    for (int j = 0; j < bench_mode; j++)
      blocks[j] = malloc(8);
    for (int j = 0; j < bench_mode; j++)
      free(blocks[j]);
  }
  delete[] blocks;
}

void bench() { start_thread_group(bench_nthread, thread); }

// CHECK: DONE
