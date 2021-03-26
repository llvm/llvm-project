// RUN: %clangxx_hwasan -O0 %s -o %t
// RUN: %env_hwasan_opts=allocator_may_return_null=0 not %run %t m1 2>&1 | FileCheck %s
// RUN: %env_hwasan_opts=allocator_may_return_null=1     %run %t m1 2>&1 | FileCheck %s --check-prefix=CHECK-NULL
// RUN: %env_hwasan_opts=allocator_may_return_null=0 not %run %t psm1 2>&1 | FileCheck %s
// RUN: %env_hwasan_opts=allocator_may_return_null=1     %run %t psm1 2>&1 | FileCheck %s --check-prefix=CHECK-NULL

// UNSUPPORTED: android

// REQUIRES: stable-runtime

// Checks that pvalloc overflows are caught. If the allocator is allowed to
// return null, the errno should be set to ENOMEM.

#include <assert.h>
#include <errno.h>
#include <malloc.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
  assert(argc == 2);
  const char *action = argv[1];

  const size_t page_size = sysconf(_SC_PAGESIZE);

  void *p = nullptr;
  if (!strcmp(action, "m1")) {
    p = pvalloc((uintptr_t)-1);
  } else if (!strcmp(action, "psm1")) {
    p = pvalloc((uintptr_t)-(page_size - 1));
  } else {
    assert(0);
  }

  fprintf(stderr, "errno: %d\n", errno);

  return p != nullptr;
}

// CHECK: {{ERROR: HWAddressSanitizer: pvalloc parameters overflow: size .* rounded up to system page size .* cannot be represented in type size_t}}
// CHECK: {{#0 0x.* in .*pvalloc}}
// CHECK: {{#1 0x.* in main .*pvalloc-overflow.cpp:}}
// CHECK: SUMMARY: HWAddressSanitizer: pvalloc-overflow

// CHECK-NULL: errno: 12
