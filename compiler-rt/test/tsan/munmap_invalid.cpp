// RUN: %clang_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s

#include "test.h"
#include <sys/mman.h>

int main() {
  // These bogus munmap's must not crash tsan runtime.
  munmap(0, 1);
  munmap(0, -1);
  munmap((void *)main, -1);
  void *p =
      mmap(0, 4096, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
  munmap(p, (1ull << 60));
  munmap(p, -10000);
  munmap(p, 0);
  fprintf(stderr, "DONE\n");
  return 0;
}

// CHECK: DONE
