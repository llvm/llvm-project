// RUN: %clangxx_tsan %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s

// bench.h needs pthread barriers which are not available on OS X
// UNSUPPORTED: darwin

#include "bench.h"
#include <memory.h>

void thread(int tid) {
  volatile long x = 0;
  switch (bench_mode) {
  case 0:
    for (int i = 0; i < bench_niter; i++)
      *(volatile char *)&x = 1;
    break;
  case 1:
    for (int i = 0; i < bench_niter; i++)
      *(volatile short *)&x = 1;
    break;
  case 2:
    for (int i = 0; i < bench_niter; i++)
      *(volatile int *)&x = 1;
    break;
  case 3:
    for (int i = 0; i < bench_niter; i++)
      *(volatile long *)&x = 1;
    break;
  case 4:
    for (int i = 0; i < bench_niter; i++)
      *(volatile char *)&x;
    break;
  case 5:
    for (int i = 0; i < bench_niter; i++)
      *(volatile short *)&x;
    break;
  case 6:
    for (int i = 0; i < bench_niter; i++)
      *(volatile int *)&x;
    break;
  case 7:
    for (int i = 0; i < bench_niter; i++)
      *(volatile long *)&x;
  case 8:
    for (int i = 0; i < bench_niter / 10; i++) {
      ((volatile long *)&x)[0];
      ((volatile int *)&x)[0];
      ((volatile short *)&x)[2];
      ((volatile char *)&x)[6];
      ((volatile char *)&x)[7];
      ((volatile long *)&x)[0] = 1;
      ((volatile int *)&x)[0] = 1;
      ((volatile short *)&x)[2] = 1;
      ((volatile char *)&x)[6] = 1;
      ((volatile char *)&x)[7] = 1;
    }
    break;
  case 9: {
    volatile long size = sizeof(x);
    for (int i = 0; i < bench_niter; i++)
      memset((void *)&x, i, size);
    break;
  }
  case 10: {
    volatile long data[2] = {};
    volatile long size = sizeof(data) - 2;
    for (int i = 0; i < bench_niter; i++)
      memset(((char *)data) + 1, i, size);
    break;
  }
  case 11: {
    volatile long data[2] = {};
    for (int i = 0; i < bench_niter / 8 / 3; i++) {
      for (int off = 0; off < 8; off++) {
        __sanitizer_unaligned_store16(((char *)data) + off, i);
        __sanitizer_unaligned_store32(((char *)data) + off, i);
        __sanitizer_unaligned_store64(((char *)data) + off, i);
      }
    }
    break;
  }
  }
}

void bench() {
  start_thread_group(bench_nthread, thread);
}

// CHECK: DONE
