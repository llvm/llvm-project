// RUN: %clang_scudo %s -o %t
// RUN: SCUDO_OPTIONS="QuarantineSizeMb=0:ThreadLocalQuarantineSizeKb=0" %run %t zeroquarantine 2>&1
// RUN: SCUDO_OPTIONS=QuarantineSizeMb=1                                 %run %t smallquarantine 2>&1

// Tests that the quarantine prevents a chunk from being reused right away.
// Also tests that a chunk will eventually become available again for
// allocation when the recycling criteria has been met.

#include <assert.h>
#include <malloc.h>
#include <stdlib.h>
#include <string.h>

#include <sanitizer/allocator_interface.h>

int main(int argc, char **argv)
{
  void *p, *old_p;
  size_t allocated_bytes, size = 1U << 16;

  assert(argc == 2);

  if (!strcmp(argv[1], "zeroquarantine")) {
    // Verifies that a chunk is deallocated right away when the local and
    // global quarantine sizes are 0.
    allocated_bytes = __sanitizer_get_current_allocated_bytes();
    p = malloc(size);
    assert(p);
    assert(__sanitizer_get_current_allocated_bytes() > allocated_bytes);
    free(p);
    assert(__sanitizer_get_current_allocated_bytes() == allocated_bytes);
  }
  if (!strcmp(argv[1], "smallquarantine")) {
    // The delayed freelist will prevent a chunk from being available right
    // away.
    p = malloc(size);
    assert(p);
    old_p = p;
    free(p);
    p = malloc(size);
    assert(p);
    assert(old_p != p);
    free(p);

    // Eventually the chunk should become available again.
    bool found = false;
    for (int i = 0; i < 0x100 && found == false; i++) {
      p = malloc(size);
      assert(p);
      found = (p == old_p);
      free(p);
    }
    assert(found == true);
  }

  return 0;
}
