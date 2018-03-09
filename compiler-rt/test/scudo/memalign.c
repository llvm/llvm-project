// RUN: %clang_scudo %s -o %t
// RUN:                                                 %run %t valid   2>&1
// RUN:                                             not %run %t invalid 2>&1
// RUN: %env_scudo_opts=allocator_may_return_null=1     %run %t invalid 2>&1

// Tests that the various aligned allocation functions work as intended. Also
// tests for the condition where the alignment is not a power of 2.

#include <assert.h>
#include <errno.h>
#include <malloc.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// Sometimes the headers may not have this...
void *aligned_alloc(size_t alignment, size_t size);

int main(int argc, char **argv)
{
  void *p = NULL;
  size_t alignment = 1U << 12;
  size_t size = 1U << 12;
  int err;

  assert(argc == 2);

  if (!strcmp(argv[1], "valid")) {
    posix_memalign(&p, alignment, size);
    assert(p);
    assert(((uintptr_t)p & (alignment - 1)) == 0);
    free(p);
    p = aligned_alloc(alignment, size);
    assert(p);
    assert(((uintptr_t)p & (alignment - 1)) == 0);
    free(p);
    // Tests various combinations of alignment and sizes
    for (int i = (sizeof(void *) == 4) ? 3 : 4; i < 19; i++) {
      alignment = 1U << i;
      for (int j = 1; j < 33; j++) {
        size = 0x800 * j;
        for (int k = 0; k < 3; k++) {
          p = memalign(alignment, size - (2 * sizeof(void *) * k));
          assert(p);
          assert(((uintptr_t)p & (alignment - 1)) == 0);
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
        assert(((uintptr_t)p & (alignment - 1)) == 0);
        free(p);
      }
    }
  }
  if (!strcmp(argv[1], "invalid")) {
    // Alignment is not a power of 2.
    p = memalign(alignment - 1, size);
    assert(!p);
    // Size is not a multiple of alignment.
    p = aligned_alloc(alignment, size >> 1);
    assert(!p);
    void *p_unchanged = (void *)0x42UL;
    p = p_unchanged;
    // Alignment is not a power of 2.
    err = posix_memalign(&p, 3, size);
    assert(p == p_unchanged);
    assert(err == EINVAL);
    // Alignment is a power of 2, but not a multiple of size(void *).
    err = posix_memalign(&p, 2, size);
    assert(p == p_unchanged);
    assert(err == EINVAL);
  }
  return 0;
}
