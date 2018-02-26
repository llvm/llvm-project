// Test max_redzone runtime option.

// RUN: %clangxx_asan -O0 %s -o %t && %env_asan_opts=max_redzone=32 %run %t 0 2>&1
// RUN: %clangxx_asan -O0 %s -o %t && %run %t 1 2>&1
// RUN: %clangxx_asan -O3 %s -o %t && %env_asan_opts=max_redzone=32 %run %t 0 2>&1
// RUN: %clangxx_asan -O3 %s -o %t && %run %t 1 2>&1

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sanitizer/allocator_interface.h>

int main(int argc, char **argv) {
  if (argc < 2)
    return 1;
  bool large_redzone = atoi(argv[1]);
  size_t before = __sanitizer_get_heap_size();
  void *pp[10000];
  for (int i = 0; i < 10000; ++i)
    pp[i] = malloc(4096 - 64);
  size_t after = __sanitizer_get_heap_size();
  for (int i = 0; i < 10000; ++i)
    free(pp[i]);
  size_t diff = after - before;
  return !(large_redzone ? diff > 46000000 : diff < 46000000);
}
