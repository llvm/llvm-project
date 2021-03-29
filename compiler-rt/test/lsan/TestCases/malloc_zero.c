// RUN: %clang_lsan %s -o %t
// RUN: %env_lsan_opts=use_stacks=0 not %run %t 2>&1 | FileCheck %s

#include <stdio.h>
#include <stdlib.h>

// CHECK: {{Leak|Address}}Sanitizer: detected memory leaks
// CHECK: {{Leak|Address}}Sanitizer: 1 byte(s) leaked in 1 allocation(s).

int main() {
  // The behavior of malloc(0) is implementation-defined.
  char *p = malloc(0);
  fprintf(stderr, "zero: %p\n", p);
  p = 0;
}
