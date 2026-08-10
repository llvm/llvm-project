// RUN: %clangxx_asan %s -o %t
// RUN: %env_asan_opts=quarantine_size_mb=2 %run %t 1
// RUN: %env_asan_opts=quarantine_size_mb=2 %run %t 4

#include <cassert>
#include <sanitizer/allocator_interface.h>
#include <stdio.h>
#include <stdlib.h>

void *g;

int main(int argc, char **argv) {
  int s = atoi(argv[1]) * 1024 * 1024;
  int a = __sanitizer_get_heap_size();
  g = malloc(s);
  int b = __sanitizer_get_heap_size();
  assert(b - a > s / 2);
  free(g);
  int c = __sanitizer_get_heap_size();
  fprintf(stderr, "%d %d\n", a, c);
  if (atoi(argv[1]) == 1)
    assert(c - a > s / 2); // NODRAIN
  else
    assert(c - a < s / 2); // DRAIN
}
