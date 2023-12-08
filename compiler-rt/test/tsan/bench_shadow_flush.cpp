// RUN: %clangxx_tsan %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s

#include <pthread.h>
#include <stdlib.h>
#include <stddef.h>
#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/mman.h>

const long kSmallPage = 4 << 10;
const long kLargePage = 2 << 20;
const long kStride    = 1 << 10;

typedef unsigned long uptr;

int main(int argc, const char **argv) {
  uptr mem_size = 4 << 20;
  if (argc > 1)
    mem_size = (uptr)atoi(argv[1]) << 20;
  uptr stride = kSmallPage;
  if (argc > 2)
    stride = (uptr)atoi(argv[2]) << 10;
  int niter = 1;
  if (argc > 3)
    niter = atoi(argv[3]);
  int stride2 = 1;
  if (argc > 4)
    stride2 = atoi(argv[4]);

  uptr sz = mem_size + stride2 * kStride + kLargePage;
  void *p = mmap(0, sz, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
  uptr a = ((uptr)p + kLargePage - 1) & ~(kLargePage - 1);
  volatile char *mem = (volatile char *)a;

  for (int i = 0; i < niter; i++) {
    for (uptr off = 0; off < mem_size; off += stride) {
      for (uptr off2 = 0; off2 < stride2; off2++)
        mem[off + off2 * kStride] = 42;
    }
  }

  fprintf(stderr, "DONE\n");
}

// CHECK: DONE

