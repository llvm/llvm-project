// RUN: %clangxx %collect_stack_traces -O0 %s -o %t

// Alignment is not a power of two:
// RUN: %env_tool_opts=allocator_may_return_null=0 not %run %t 17 2>&1 | FileCheck %s
// Alignment is not a power of two, although is a multiple of sizeof(void*):
// RUN: %env_tool_opts=allocator_may_return_null=0 not %run %t 24 2>&1 | FileCheck %s
// Alignment is not a multiple of sizeof(void*), although is a power of 2:
// RUN: %env_tool_opts=allocator_may_return_null=0 not %run %t 2 2>&1 | FileCheck %s
// Alignment is 0:
// RUN: %env_tool_opts=allocator_may_return_null=0 not %run %t 0 2>&1 | FileCheck %s

// The same for allocator_may_return_null=1:
// RUN: %env_tool_opts=allocator_may_return_null=1 %run %t 17 2>&1 | FileCheck %s --check-prefix=CHECK-NULL
// RUN: %env_tool_opts=allocator_may_return_null=1 %run %t 24 2>&1 | FileCheck %s --check-prefix=CHECK-NULL
// RUN: %env_tool_opts=allocator_may_return_null=1 %run %t 2 2>&1 | FileCheck %s --check-prefix=CHECK-NULL
// RUN: %env_tool_opts=allocator_may_return_null=1 %run %t 0 2>&1 | FileCheck %s --check-prefix=CHECK-NULL

// REQUIRES: stable-runtime

// UNSUPPORTED: ubsan

#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  assert(argc == 2);
  const int alignment = atoi(argv[1]);

  void* const kInitialPtrValue = reinterpret_cast<void*>(0x2a);
  void *p = kInitialPtrValue;

  errno = 0;
  int res = posix_memalign(&p, alignment, 100);
  // CHECK: {{ERROR: .*Sanitizer: invalid alignment requested in posix_memalign}}
  // CHECK: {{#0 .*posix_memalign}}
  // CHECK: {{#1 .*main .*posix_memalign-alignment.cc:}}[[@LINE-3]]
  // CHECK: {{SUMMARY: .*Sanitizer: invalid-posix-memalign-alignment}}

  // The NULL pointer is printed differently on different systems, while (long)0
  // is always the same.
  fprintf(stderr, "errno: %d, res: %d, p: %lx\n", errno, res, (long)p);
  // CHECK-NULL: errno: 0, res: 22, p: 2a

  return 0;
}
