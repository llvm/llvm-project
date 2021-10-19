// RUN: %clang_lsan %s -o %t
// RUN: %run %t

#include <assert.h>
#include <stdlib.h>

int main() {
  char *p = malloc(1);
  // The behavior of realloc(p, 0) is implementation-defined.
  // We free the allocation.
  assert(realloc(p, 0) == NULL);
  p = 0;
}
