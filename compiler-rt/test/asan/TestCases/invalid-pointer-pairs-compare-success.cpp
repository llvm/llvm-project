// RUN: %clangxx_asan -O0 %s -o %t -mllvm -asan-detect-invalid-pointer-pair

// RUN: %env_asan_opts=detect_invalid_pointer_pairs=2 %run %t

#include <assert.h>
#include <stdlib.h>

int foo(char *p) {
  char *p2 = p + 20;
  return p > p2;
}

int bar(char *p, char *q) {
  return p <= q;
}

int baz(char *p, char *q) {
  return p != 0 && p < q;
}

char global[8192] = {};
char small_global[7] = {};

int main() {
  // Heap allocated memory.
  char *p = (char *)malloc(42);
  int r = foo(p);
  free(p);

  p = (char *)malloc(1024);
  bar(p, p + 1024);
  bar(p + 1024, p + 1023);
  bar(p + 1, p + 1023);
  free(p);

  p = (char *)malloc(4096);
  bar(p, p + 4096);
  bar(p + 10, p + 100);
  bar(p + 1024, p + 4096);
  bar(p + 4095, p + 4096);
  bar(p + 4095, p + 4094);
  bar(p + 100, p + 4096);
  bar(p + 100, p + 4094);
  free(p);

  // Global variable.
  bar(&global[0], &global[1]);
  bar(&global[1], &global[2]);
  bar(&global[2], &global[1]);
  bar(&global[0], &global[100]);
  bar(&global[1000], &global[7000]);
  bar(&global[500], &global[10]);
  p = &global[0];
  bar(p, p + 8192);
  p = &global[8000];
  bar(p, p + 192);

  p = &small_global[0];
  bar(p, p + 1);
  bar(p, p + 7);
  bar(p + 7, p + 1);
  bar(p + 6, p + 7);
  bar(p + 7, p + 7);

  // Stack variable.
  char stack[10000];
  bar(&stack[0], &stack[100]);
  bar(&stack[1000], &stack[9000]);
  bar(&stack[500], &stack[10]);

  baz(0, &stack[10]);

  return 0;
}
