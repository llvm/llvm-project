// RUN: %clangxx_msan -fsanitize-memory-track-origins -O0 %s -o %t && not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out
// RUN: %clangxx_msan -fsanitize-memory-track-origins -O2 %s -o %t && not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

// This test relies on realloc from 100 to 101 being done in-place.

#include <stdlib.h>
int main(int argc, char **argv) {
  char *p = (char *)malloc(100);
  p = (char *)realloc(p, 101);
  char x = p[100];
  free(p);
  return x;
  // CHECK: WARNING: MemorySanitizer: use-of-uninitialized-value
  // CHECK: {{#0 0x.* in main .*realloc-origin.cpp:}}[[@LINE-2]]

  // CHECK: Uninitialized value was created by a heap allocation
  // CHECK: {{#0 0x.* in .*realloc}}
  // CHECK: {{#1 0x.* in main .*realloc-origin.cpp:}}[[@LINE-9]]
}
