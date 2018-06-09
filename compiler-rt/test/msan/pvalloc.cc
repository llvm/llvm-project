// RUN: %clangxx_msan -fsanitize-memory-track-origins -O0 -g %s -o %t
// RUN: MSAN_OPTIONS=allocator_may_return_null=0 not %run %t m1 2>&1 | FileCheck %s
// RUN: MSAN_OPTIONS=allocator_may_return_null=1     %run %t m1 2>&1
// RUN: MSAN_OPTIONS=allocator_may_return_null=0 not %run %t psm1 2>&1 | FileCheck %s
// RUN: MSAN_OPTIONS=allocator_may_return_null=1     %run %t psm1 2>&1

// pvalloc is Linux only
// UNSUPPORTED: win32, freebsd, netbsd

// Checks that pvalloc overflows are caught. If the allocator is allowed to
// return null, the errno should be set to ENOMEM.

#include <assert.h>
#include <errno.h>
#include <malloc.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
  void *p;
  size_t page_size;

  assert(argc == 2);

  page_size = sysconf(_SC_PAGESIZE);
  // Check that the page size is a power of two.
  assert((page_size & (page_size - 1)) == 0);

  if (!strcmp(argv[1], "m1")) {
    p = pvalloc((uintptr_t)-1);
    assert(!p);
    assert(errno == ENOMEM);
  }
  if (!strcmp(argv[1], "psm1")) {
    p = pvalloc((uintptr_t)-(page_size - 1));
    assert(!p);
    assert(errno == ENOMEM);
  }

  return 0;
}

// CHECK: {{ERROR: MemorySanitizer: pvalloc parameters overflow: size .* rounded up to system page size .* cannot be represented in type size_t}}
// CHECK: {{#0 0x.* in .*pvalloc}}
// CHECK: SUMMARY: MemorySanitizer: pvalloc-overflow
