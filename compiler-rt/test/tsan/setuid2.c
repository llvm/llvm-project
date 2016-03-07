// RUN: %clang_tsan -O1 %s -o %t && %env_tsan_opts=flush_memory_ms=1:memory_limit_mb=1 %run %t 2>&1 | FileCheck %s
//
// setuid(0) hangs on powerpc64 big endian.  When this is fixed remove
// the unsupported flag.
// https://llvm.org/bugs/show_bug.cgi?id=25799
//
// UNSUPPORTED: powerpc64-unknown-linux-gnu
#include "test.h"
#include <sys/types.h>
#include <unistd.h>
#include <time.h>

// Test that setuid call works in presence of stoptheworld.

int main() {
  unsigned long long tp0, tp1;
  tp0 = monotonic_clock_ns();
  tp1 = monotonic_clock_ns();
  while (tp1 - tp0 < 3 * 1000000000ull) {
    tp1 = monotonic_clock_ns();
    setuid(0);
  }
  fprintf(stderr, "DONE\n");
  return 0;
}

// CHECK: DONE
