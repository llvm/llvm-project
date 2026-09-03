// RUN: %clang_dsan %s -o %t
// RUN: %env_dsan_opts=allocator_may_return_null=1 %run %t

#include <stdint.h>
#include <stdlib.h>

int main(void) {
  char *p = malloc(16);
  p[0] = 42;
  if (realloc(p, SIZE_MAX) != NULL)
    return 1;
  if (p[0] != 42)
    return 2;
  free(p);
  return 0;
}
