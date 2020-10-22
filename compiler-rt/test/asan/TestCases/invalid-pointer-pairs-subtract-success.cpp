// RUN: %clangxx_asan -O0 %s -o %t -mllvm -asan-detect-invalid-pointer-pair

// RUN: %env_asan_opts=detect_invalid_pointer_pairs=2 %run %t
// RUN: %env_asan_opts=detect_invalid_pointer_pairs=2,detect_stack_use_after_return=1 %run %t

#include <assert.h>
#include <stdlib.h>

int bar(char *p, char *q) {
  return p - q;
}

char global[10000] = {};

int main() {
  // Heap allocated memory.
  char *p = (char *)malloc(42);
  int r = bar(p, p + 20);
  free(p);

  // Global variable.
  bar(&global[0], &global[100]);
  bar(&global[1000], &global[9000]);
  bar(&global[500], &global[10]);
  bar(&global[0], &global[10000]);

  // Stack variable.
  char stack[10000];
  bar(&stack[0], &stack[100]);
  bar(&stack[1000], &stack[9000]);
  bar(&stack[500], &stack[10]);

  return 0;
}
