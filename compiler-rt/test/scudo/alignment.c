// RUN: %clang_scudo %s -o %t
// RUN: not %run %t pointers 2>&1 | FileCheck %s

// Tests that a non MinAlignment aligned pointer will trigger the associated
// error on deallocation.

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
  assert(argc == 2);
  if (!strcmp(argv[1], "pointers")) {
    void *p = malloc(1U << 16);
    assert(p);
    free((void *)((uintptr_t)p | 1));
  }
  return 0;
}

// CHECK: ERROR: misaligned pointer when deallocating address
