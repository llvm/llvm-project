// RUN: %clang_scudo %s -o %t
// RUN:                                                 %run %t valid   2>&1
// RUN:                                             not %run %t invalid 2>&1
// RUN: %env_scudo_opts=allocator_may_return_null=1     %run %t invalid 2>&1
// UNSUPPORTED: android

// Tests that valloc and pvalloc work as intended.

#include <assert.h>
#include <errno.h>
#include <malloc.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

size_t round_up_to(size_t size, size_t alignment) {
  return (size + alignment - 1) & ~(alignment - 1);
}

int main(int argc, char **argv)
{
  void *p = NULL;
  size_t size, page_size;

  assert(argc == 2);

  page_size = sysconf(_SC_PAGESIZE);
  // Check that the page size is a power of two.
  assert((page_size & (page_size - 1)) == 0);

  if (!strcmp(argv[1], "valid")) {
    for (int i = (sizeof(void *) == 4) ? 3 : 4; i < 21; i++) {
      size = 1U << i;
      p = valloc(size - (2 * sizeof(void *)));
      assert(p);
      assert(((uintptr_t)p & (page_size - 1)) == 0);
      free(p);
      p = pvalloc(size - (2 * sizeof(void *)));
      assert(p);
      assert(((uintptr_t)p & (page_size - 1)) == 0);
      assert(malloc_usable_size(p) >= round_up_to(size, page_size));
      free(p);
      p = valloc(size);
      assert(p);
      assert(((uintptr_t)p & (page_size - 1)) == 0);
      free(p);
      p = pvalloc(size);
      assert(p);
      assert(((uintptr_t)p & (page_size - 1)) == 0);
      assert(malloc_usable_size(p) >= round_up_to(size, page_size));
      free(p);
    }
  }
  if (!strcmp(argv[1], "invalid")) {
    // Size passed to pvalloc overflows when rounded up.
    p = pvalloc((size_t)-1);
    assert(!p);
    assert(errno == ENOMEM);
    errno = 0;
    p = pvalloc((size_t)-page_size);
    assert(!p);
    assert(errno == ENOMEM);
  }
  return 0;
}
