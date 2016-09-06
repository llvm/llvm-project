// RUN: %clang_scudo %s -o %t
// RUN: %run %t 2>&1

// Tests that a regular workflow of allocation, memory fill and free works as
// intended. Also tests that a zero-sized allocation succeeds.

#include <malloc.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
  void *p;
  size_t size = 1U << 8;

  p = malloc(size);
  if (!p)
    return 1;
  memset(p, 'A', size);
  free(p);
  p = malloc(0);
  if (!p)
    return 1;
  free(p);

  return 0;
}
