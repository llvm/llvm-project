// RUN: %clang_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s

// Fails on Darwin bots:
// https://green.lab.llvm.org/green//job/clang-stage1-RA/25954/consoleFull
// and on clang-s390x-linux-lnt:
// https://lab.llvm.org/buildbot#builders/45/builds/5224
// Presumably the test is not 100% legal and kernel is allowed
// to unmap part of the range (e.g. .text) and then fail.
// So let's be conservative:
// REQUIRES: linux, x86_64-target-arch

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
