// Check that __asan_poison_memory_region and ASAN_OPTIONS=poison_history_size work for partial granules.
//
// RUN: %clangxx_asan -O0 %s -o %t && env ASAN_OPTIONS=poison_history_size=1000 not %run %t 10 20 10 2>&1 | FileCheck %s
//
// Partial granule
// RUN: %clangxx_asan -O0 %s -o %t && env ASAN_OPTIONS=poison_history_size=1000 not %run %t 10 20 20 2>&1 | FileCheck %s

// TODO
// REQUIRES: linux
// UNSUPPORTED: android

#include <cassert>
#include <stdio.h>
#include <stdlib.h>

#include <sanitizer/asan_interface.h>

void honey_ive_poisoned_the_memory(char *x, size_t poison_offset,
                                   size_t poison_size) {
  __asan_poison_memory_region(x + poison_offset, poison_size);
}

void foo(char *x, size_t poison_offset, size_t poison_size) {
  honey_ive_poisoned_the_memory(x, poison_offset, poison_size);
}

int main(int argc, char **argv) {
  assert(argc > 3);
  size_t poison_offset = atoi(argv[1]);
  size_t poison_size = atoi(argv[2]);
  size_t access_offset = atoi(argv[3]);
  char *x = new char[64];
  x[10] = 0;
  foo(x, poison_offset, poison_size);
  // Bytes [0,   9]: addressable
  // Bytes [10,  31]: poisoned by A
  // Bytes [32,  63]: addressable

  int res = x[access_offset]; // BOOOM
  // CHECK: ERROR: AddressSanitizer: use-after-poison
  // CHECK: main{{.*}}use-after-poison-history-size-partial-granule.cpp:[[@LINE-2]]

  // CHECK: Memory was manually poisoned by thread T0:
  // CHECK: honey_ive_poisoned_the_memory{{.*}}use-after-poison-history-size-partial-granule.cpp:[[@LINE-24]]
  // CHECK: foo{{.*}}use-after-poison-history-size-partial-granule.cpp:[[@LINE-21]]
  // CHECK: main{{.*}}use-after-poison-history-size-partial-granule.cpp:[[@LINE-12]]

  delete[] x;

  return 0;
}
