// RUN: %clangxx_scudo %s -lstdc++ -o %t
// RUN: %env_scudo_opts=allocator_may_return_null=0 not %run %t malloc 2>&1 | FileCheck %s
// RUN: %env_scudo_opts=allocator_may_return_null=1     %run %t malloc 2>&1
// RUN: %env_scudo_opts=allocator_may_return_null=0 not %run %t calloc 2>&1 | FileCheck %s
// RUN: %env_scudo_opts=allocator_may_return_null=1     %run %t calloc 2>&1
// RUN: %env_scudo_opts=allocator_may_return_null=0 not %run %t new 2>&1 | FileCheck %s
// RUN: %env_scudo_opts=allocator_may_return_null=1 not %run %t new 2>&1 | FileCheck %s
// RUN: %env_scudo_opts=allocator_may_return_null=0 not %run %t new-nothrow 2>&1 | FileCheck %s
// RUN: %env_scudo_opts=allocator_may_return_null=1     %run %t new-nothrow 2>&1
// RUN:                                                 %run %t usable 2>&1

// Tests for various edge cases related to sizes, notably the maximum size the
// allocator can allocate. Tests that an integer overflow in the parameters of
// calloc is caught.

#include <assert.h>
#include <malloc.h>
#include <stdlib.h>
#include <string.h>

#include <limits>
#include <new>

int main(int argc, char **argv) {
  assert(argc == 2);
  const char *action = argv[1];
  fprintf(stderr, "%s:\n", action);

#if __LP64__ || defined(_WIN64)
  static const size_t kMaxAllowedMallocSize = 1ULL << 40;
  static const size_t kChunkHeaderSize = 16;
#else
  static const size_t kMaxAllowedMallocSize = 2UL << 30;
  static const size_t kChunkHeaderSize = 8;
#endif

  if (!strcmp(action, "malloc")) {
    void *p = malloc(kMaxAllowedMallocSize);
    assert(!p);
    p = malloc(kMaxAllowedMallocSize - kChunkHeaderSize);
    assert(!p);
  } else if (!strcmp(action, "calloc")) {
    // Trigger an overflow in calloc.
    size_t size = std::numeric_limits<size_t>::max();
    void *p = calloc((size / 0x1000) + 1, 0x1000);
    assert(!p);
  } else if (!strcmp(action, "new")) {
    void *p = operator new(kMaxAllowedMallocSize);
    assert(!p);
  } else if (!strcmp(action, "new-nothrow")) {
    void *p = operator new(kMaxAllowedMallocSize, std::nothrow);
    assert(!p);
  } else if (!strcmp(action, "usable")) {
    // Playing with the actual usable size of a chunk.
    void *p = malloc(1007);
    assert(p);
    size_t size = malloc_usable_size(p);
    assert(size >= 1007);
    memset(p, 'A', size);
    p = realloc(p, 2014);
    assert(p);
    size = malloc_usable_size(p);
    assert(size >= 2014);
    memset(p, 'B', size);
    free(p);
  } else {
    assert(0);
  }

  return 0;
}

// CHECK: allocator is terminating the process
