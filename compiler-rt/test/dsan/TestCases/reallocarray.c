// REQUIRES: linux
// RUN: %clang_dsan %s -o %t
// RUN: %env_dsan_opts=allocator_may_return_null=1 %run %t

#define _GNU_SOURCE
#include <stdint.h>
#include <stdlib.h>

int main(void) {
  char *p = malloc(16);
  p[0] = 42;
  if (reallocarray(p, SIZE_MAX, 2) != NULL)
    return 1;
  if (p[0] != 42)
    return 2;
  free(p);
  return 0;
}
