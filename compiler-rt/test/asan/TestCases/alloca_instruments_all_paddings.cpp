// RUN: %clangxx_asan -O0 -mllvm -asan-instrument-dynamic-allocas %s -o %t
// RUN: %clangxx_asan -O3 -mllvm -asan-instrument-dynamic-allocas %s -o %t
// RUN: %run %t 2>&1
//

#include "sanitizer/asan_interface.h"
#include <assert.h>

__attribute__((noinline)) void foo(int index, int len) {
  volatile char str[len] __attribute__((aligned(32)));
  assert(!(reinterpret_cast<long>(str) & 31L));
  char *q = (char *)__asan_region_is_poisoned((char *)str, 64);
  assert(q && ((q - str) == index));
}

int main(int argc, char **argv) {
  for (int i = 1; i < 33; ++i)
    foo(i, i);

  for (int i = 1; i < 33; ++i)
    foo(i, i);

  return 0;
}
