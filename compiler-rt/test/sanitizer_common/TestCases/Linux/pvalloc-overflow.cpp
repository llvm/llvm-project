// RUN: %clangxx %collect_stack_traces -O0 %s -o %t
// RUN: %env_tool_opts=allocator_may_return_null=0 not %run %t m1 2>&1 | FileCheck %s
// RUN: %env_tool_opts=allocator_may_return_null=1     %run %t m1 2>&1 | FileCheck %s --check-prefix=CHECK-NULL
// RUN: %env_tool_opts=allocator_may_return_null=0 not %run %t psm1 2>&1 | FileCheck %s
// RUN: %env_tool_opts=allocator_may_return_null=1     %run %t psm1 2>&1 | FileCheck %s --check-prefix=CHECK-NULL

// REQUIRES: stable-runtime

// UNSUPPORTED: android, freebsd, netbsd, ubsan

// Checks that pvalloc overflows are caught. If the allocator is allowed to
// return null, the errno should be set to ENOMEM.

#include <assert.h>
#include <errno.h>
#include <malloc.h>
#include <stdio.h>
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
  // CHECK: {{ERROR: .*Sanitizer: pvalloc parameters overflow: size .* rounded up to system page size .* cannot be represented in type size_t}}
  // CHECK: {{#0 .*pvalloc}}
  // CHECK: {{#[12] .*main .*pvalloc-overflow.cpp:}}
  // CHECK: {{SUMMARY: .*Sanitizer: pvalloc-overflow}}

  // The NULL pointer is printed differently on different systems, while (long)0
  // is always the same.
  fprintf(stderr, "errno: %d, p: %lx\n", errno, (long)p);
  // CHECK-NULL: errno: 12, p: 0

  return 0;
}
