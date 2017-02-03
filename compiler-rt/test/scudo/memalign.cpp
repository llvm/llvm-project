// RUN: %clang_scudo %s -o %t
// RUN:     %run %t valid   2>&1
// RUN: not %run %t invalid 2>&1 | FileCheck %s

// Tests that the various aligned allocation functions work as intended. Also
// tests for the condition where the alignment is not a power of 2.

#include <assert.h>
#include <malloc.h>
#include <stdlib.h>
#include <string.h>

// Reduce the size of the quarantine, or the test can run out of aligned memory
// on 32-bit for the larger alignments.
extern "C" const char *__scudo_default_options() {
  return "QuarantineSizeMb=1";
}

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
    assert(p);
    free(p);
    p = aligned_alloc(alignment, size);
    assert(p);
    free(p);
    // Tests various combinations of alignment and sizes
    for (int i = (sizeof(void *) == 4) ? 3 : 4; i < 19; i++) {
      alignment = 1U << i;
      for (int j = 1; j < 33; j++) {
        size = 0x800 * j;
        for (int k = 0; k < 3; k++) {
          p = memalign(alignment, size - (2 * sizeof(void *) * k));
          assert(p);
          free(p);
        }
      }
    }
    // For larger alignment, reduce the number of allocations to avoid running
    // out of potential addresses (on 32-bit).
    for (int i = 19; i <= 24; i++) {
      for (int k = 0; k < 3; k++) {
        p = memalign(alignment, 0x1000 - (2 * sizeof(void *) * k));
        assert(p);
        free(p);
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
