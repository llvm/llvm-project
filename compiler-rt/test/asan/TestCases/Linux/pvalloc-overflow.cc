// RUN: %clangxx_asan  %s -o %t
// RUN: ASAN_OPTIONS=allocator_may_return_null=0 not %run %t m1 2>&1 | FileCheck %s
// RUN: ASAN_OPTIONS=allocator_may_return_null=1     %run %t m1 2>&1
// RUN: ASAN_OPTIONS=allocator_may_return_null=0 not %run %t psm1 2>&1 | FileCheck %s
// RUN: ASAN_OPTIONS=allocator_may_return_null=1     %run %t psm1 2>&1

// UNSUPPORTED: freebsd, android

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

// CHECK: AddressSanitizer's allocator is terminating the process
