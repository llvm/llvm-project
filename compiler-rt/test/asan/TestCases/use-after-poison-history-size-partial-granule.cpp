// Check that __asan_poison_memory_region and ASAN_OPTIONS=poison_history_size work for partial granules.
//
// RUN: %clangxx_asan -O0 %s -o %t && env ASAN_OPTIONS=poison_history_size=1000 not %run %t 20 2>&1 | FileCheck %s
//
// Partial granule
// RUN: %clangxx_asan -O0 %s -o %t && env ASAN_OPTIONS=poison_history_size=1000 not %run %t    2>&1 | FileCheck %s

// TODO
// REQUIRES: linux
// UNSUPPORTED: android

#include <stdio.h>
#include <stdlib.h>

extern "C" void __asan_poison_memory_region(void *, size_t);
extern "C" void __asan_unpoison_memory_region(void *, size_t);

void honey_ive_poisoned_the_memory(char *x) {
  __asan_poison_memory_region(x + 10, 20);
}

void foo(char *x) { honey_ive_poisoned_the_memory(x); }

int main(int argc, char **argv) {
  char *x = new char[64];
  x[10] = 0;
  foo(x);
  // Bytes [0,   9]: addressable
  // Bytes [10,  31]: poisoned by A
  // Bytes [32,  63]: addressable

  int res = x[argc * 10]; // BOOOM
  // CHECK: ERROR: AddressSanitizer: use-after-poison
  // CHECK: main{{.*}}use-after-poison-history-size-partial-granule.cpp:[[@LINE-2]]

  // CHECK: Memory was manually poisoned by thread T0:
  // CHECK: honey_ive_poisoned_the_memory{{.*}}use-after-poison-history-size-partial-granule.cpp:[[@LINE-18]]
  // CHECK: foo{{.*}}use-after-poison-history-size-partial-granule.cpp:[[@LINE-16]]
  // CHECK: main{{.*}}use-after-poison-history-size-partial-granule.cpp:[[@LINE-12]]

  delete[] x;

  return 0;
}
