// RUN: %clang_scudo %s -o %t
// RUN: SCUDO_OPTIONS=allocator_may_return_null=0 not %run %t malloc 2>&1 | FileCheck %s
// RUN: SCUDO_OPTIONS=allocator_may_return_null=1     %run %t malloc 2>&1
// RUN: SCUDO_OPTIONS=allocator_may_return_null=0 not %run %t calloc 2>&1 | FileCheck %s
// RUN: SCUDO_OPTIONS=allocator_may_return_null=1     %run %t calloc 2>&1
// RUN:                                               %run %t usable 2>&1

// Tests for various edge cases related to sizes, notably the maximum size the
// allocator can allocate. Tests that an integer overflow in the parameters of
// calloc is caught.

#include <assert.h>
#include <malloc.h>
#include <stdlib.h>
#include <string.h>

#include <limits>

int main(int argc, char **argv)
{
  assert(argc == 2);
  if (!strcmp(argv[1], "malloc")) {
    // Currently the maximum size the allocator can allocate is 1ULL<<40 bytes.
    size_t size = std::numeric_limits<size_t>::max();
    void *p = malloc(size);
    if (p)
      return 1;
    size = (1ULL << 40) - 16;
    p = malloc(size);
    if (p)
      return 1;
  }
  if (!strcmp(argv[1], "calloc")) {
    // Trigger an overflow in calloc.
    size_t size = std::numeric_limits<size_t>::max();
    void *p = calloc((size / 0x1000) + 1, 0x1000);
    if (p)
      return 1;
  }
  if (!strcmp(argv[1], "usable")) {
    // Playing with the actual usable size of a chunk.
    void *p = malloc(1007);
    if (!p)
      return 1;
    size_t size = malloc_usable_size(p);
    if (size < 1007)
      return 1;
    memset(p, 'A', size);
    p = realloc(p, 2014);
    if (!p)
      return 1;
    size = malloc_usable_size(p);
    if (size < 2014)
      return 1;
    memset(p, 'B', size);
    free(p);
  }
  return 0;
}

// CHECK: allocator is terminating the process
