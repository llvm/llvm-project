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
  void *p = nullptr;
  size_t alignment = 1U << 12;
  size_t size = 1U << 12;

  assert(argc == 2);

  if (!strcmp(argv[1], "valid")) {
    posix_memalign(&p, alignment, size);
    if (!p)
      return 1;
    free(p);
    p = aligned_alloc(alignment, size);
    if (!p)
      return 1;
    free(p);
    // Tests various combinations of alignment and sizes
    for (int i = (sizeof(void *) == 4) ? 3 : 4; i <= 24; i++) {
      alignment = 1U << i;
      for (int j = 1; j < 33; j++) {
        size = 0x800 * j;
        for (int k = 0; k < 3; k++) {
          p = memalign(alignment, size - (16 * k));
          if (!p)
            return 1;
          free(p);
        }
      }
    }
  }
  if (!strcmp(argv[1], "invalid")) {
    p = memalign(alignment - 1, size);
    free(p);
  }
  return 0;
}

// CHECK: ERROR: alignment is not a power of 2
