// RUN: %clang_scudo %s -o %t
// RUN:     %run %t valid   2>&1
// RUN: not %run %t invalid 2>&1 | FileCheck %s

// Tests that the various aligned allocation functions work as intended. Also
// tests for the condition where the alignment is not a power of 2.

#include <assert.h>
#include <malloc.h>
#include <stdlib.h>
#include <string.h>

// Sometimes the headers may not have this...
extern "C" void *aligned_alloc (size_t alignment, size_t size);

int main(int argc, char **argv)
{
  void *p;
  size_t alignment = 1U << 12;
  size_t size = alignment;

  assert(argc == 2);
  if (!strcmp(argv[1], "valid")) {
    p = memalign(alignment, size);
    if (!p)
      return 1;
    free(p);
    p = nullptr;
    posix_memalign(&p, alignment, size);
    if (!p)
      return 1;
    free(p);
    p = aligned_alloc(alignment, size);
    if (!p)
      return 1;
    free(p);
  }
  if (!strcmp(argv[1], "invalid")) {
    p = memalign(alignment - 1, size);
    free(p);
  }
  return 0;
}

// CHECK: ERROR: malloc alignment is not a power of 2
