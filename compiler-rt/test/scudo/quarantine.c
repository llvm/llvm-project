// RUN: %clang_scudo %s -o %t
// RUN: %env_scudo_opts="QuarantineSizeMb=1:QuarantineSizeKb=64"           not %run %t unused 2>&1
// RUN: %env_scudo_opts="QuarantineSizeMb=1:QuarantineChunksUpToSize=256"  not %run %t unused 2>&1
// RUN: %env_scudo_opts="QuarantineSizeKb=0:ThreadLocalQuarantineSizeKb=0"     %run %t zeroquarantine 2>&1
// RUN: %env_scudo_opts=QuarantineSizeKb=64                                    %run %t smallquarantine 2>&1
// RUN: %env_scudo_opts=QuarantineChunksUpToSize=256                           %run %t threshold 2>&1
// RUN: %env_scudo_opts="QuarantineSizeMb=1"                                   %run %t oldquarantine 2>&1

// Tests that the quarantine prevents a chunk from being reused right away.
// Also tests that a chunk will eventually become available again for
// allocation when the recycling criteria has been met. Finally, tests the
// threshold up to which a chunk is quarantine, and the old quarantine behavior.

#include <assert.h>
#include <malloc.h>
#include <stdlib.h>
#include <string.h>

#include <sanitizer/allocator_interface.h>

int main(int argc, char **argv)
{
  void *p, *old_p;
  size_t allocated_bytes, size = 1U << 8, alignment = 1U << 8;

  assert(argc == 2);
  // First, warm up the allocator for the classes used.
  p = malloc(size);
  assert(p);
  free(p);
  p = malloc(size + 1);
  assert(p);
  free(p);
  assert(posix_memalign(&p, alignment, size) == 0);
  assert(p);
  free(p);
  assert(posix_memalign(&p, alignment, size + 1) == 0);
  assert(p);
  free(p);

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
    char found = 0;
    for (int i = 0; i < 0x200 && !found; i++) {
      p = malloc(size);
      assert(p);
      found = (p == old_p);
      free(p);
    }
    assert(found);
  }
  if (!strcmp(argv[1], "threshold")) {
    // Verifies that a chunk of size greater than the threshold will be freed
    // right away. Alignment has no impact on the threshold.
    allocated_bytes = __sanitizer_get_current_allocated_bytes();
    p = malloc(size + 1);
    assert(p);
    assert(__sanitizer_get_current_allocated_bytes() > allocated_bytes);
    free(p);
    assert(__sanitizer_get_current_allocated_bytes() == allocated_bytes);
    assert(posix_memalign(&p, alignment, size + 1) == 0);
    assert(__sanitizer_get_current_allocated_bytes() > allocated_bytes);
    free(p);
    assert(__sanitizer_get_current_allocated_bytes() == allocated_bytes);
    // Verifies that a chunk of size lower or equal to the threshold will be
    // quarantined.
    p = malloc(size);
    assert(p);
    assert(__sanitizer_get_current_allocated_bytes() > allocated_bytes);
    free(p);
    assert(__sanitizer_get_current_allocated_bytes() > allocated_bytes);
    allocated_bytes = __sanitizer_get_current_allocated_bytes();
    assert(posix_memalign(&p, alignment, size) == 0);
    assert(__sanitizer_get_current_allocated_bytes() > allocated_bytes);
    free(p);
    assert(__sanitizer_get_current_allocated_bytes() > allocated_bytes);
  }
  if (!strcmp(argv[1], "oldquarantine")) {
    // Verifies that we quarantine everything if the deprecated quarantine
    // option is specified. Alignment has no impact on the threshold.
    allocated_bytes = __sanitizer_get_current_allocated_bytes();
    p = malloc(size);
    assert(p);
    assert(__sanitizer_get_current_allocated_bytes() > allocated_bytes);
    free(p);
    assert(__sanitizer_get_current_allocated_bytes() > allocated_bytes);
    allocated_bytes = __sanitizer_get_current_allocated_bytes();
    assert(posix_memalign(&p, alignment, size) == 0);
    assert(p);
    assert(__sanitizer_get_current_allocated_bytes() > allocated_bytes);
    free(p);
    assert(__sanitizer_get_current_allocated_bytes() > allocated_bytes);
    // Secondary backed allocation.
    allocated_bytes = __sanitizer_get_current_allocated_bytes();
    p = malloc(1U << 19);
    assert(p);
    assert(__sanitizer_get_current_allocated_bytes() > allocated_bytes);
    free(p);
    assert(__sanitizer_get_current_allocated_bytes() > allocated_bytes);
  }

  return 0;
}
