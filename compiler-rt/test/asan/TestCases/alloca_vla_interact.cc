// RUN: %clangxx_asan -O0 -mllvm -asan-instrument-allocas %s -o %t
// RUN: %run %t 2>&1
//
// REQUIRES: stable-runtime

// This testcase checks correct interaction between VLAs and allocas.

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include "sanitizer/asan_interface.h"

// MSVC provides _alloca instead of alloca.
#if defined(_MSC_VER) && !defined(alloca)
# define alloca _alloca
#endif

#define RZ 32

__attribute__((noinline)) void foo(int len) {
  char *top, *bot;
  // This alloca call should live until the end of foo.
  char *alloca1 = (char *)alloca(len);
  assert(!(reinterpret_cast<uintptr_t>(alloca1) & 31L));
  // This should be first poisoned address after loop.
  top = alloca1 - RZ;
  for (int i = 0; i < 32; ++i) {
    // Check that previous alloca was unpoisoned at the end of iteration.
    if (i) assert(!__asan_region_is_poisoned(bot, 96));
    // VLA is unpoisoned at the end of iteration.
    volatile char array[i];
    assert(!(reinterpret_cast<uintptr_t>(array) & 31L));
    // Alloca is unpoisoned at the end of iteration,
    // because dominated by VLA.
    bot = (char *)alloca(i) - RZ;
  }
  // Check that all allocas from loop were unpoisoned correctly.
  void *q = __asan_region_is_poisoned(bot, (char *)top - (char *)bot + 1);
  assert(q == top);
}

int main(int argc, char **argv) {
  foo(32);
  return 0;
}
