// RUN: %clangxx_asan -O0 %s -o %t -mllvm -asan-detect-invalid-pointer-pair

// RUN: %env_asan_opts=detect_invalid_pointer_pairs=1 %run %t

#include <assert.h>
#include <stdlib.h>

int foo(char *p, char *q) {
  return p <= q;
}

char global[8192] = {};
char small_global[7] = {};

int main() {
  // Heap allocated memory.
  char *p = (char *)malloc(42);
  int r = foo(p, NULL);
  free(p);

  p = (char *)malloc(1024);
  foo(NULL, p);
  free(p);

  p = (char *)malloc(4096);
  foo(p, NULL);
  free(p);

  // Global variable.
  foo(&global[0], NULL);
  foo(&global[1000], NULL);

  p = &small_global[0];
  foo(p, NULL);

  // Stack variable.
  char stack[10000];
  foo(&stack[0], NULL);
  foo(NULL, &stack[9000]);

  return 0;
}
