// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s

// UNSUPPORTED: ios

#include "test.h"
#include <sys/mman.h>

// Test for previously unbounded memory consumption for large mallocs.
// Code allocates a large memory block (that is handled by LargeMmapAllocator),
// and forces allocation of meta shadow for the block. Then freed the block.
// But meta shadow was not unmapped. Then code occupies the virtual memory
// range of the block with something else (that does not need meta shadow).
// And repeats. As the result meta shadow growed infinitely.
// This program used to consume >2GB. Now it consumes <50MB.

int main() {
  for (int i = 0; i < 1000; i++) {
    const int kSize = 1 << 20;
    const int kPageSize = 4 << 10;
    volatile int *p = new int[kSize];
    for (int j = 0; j < kSize; j += kPageSize / sizeof(*p))
      __atomic_store_n(&p[i], 1, __ATOMIC_RELEASE);
    delete[] p;
    mmap(0, kSize * sizeof(*p) + kPageSize, PROT_NONE, MAP_PRIVATE | MAP_ANON,
        -1, 0);
  }
  fprintf(stderr, "DONE\n");
  return 0;
}

// CHECK: DONE
