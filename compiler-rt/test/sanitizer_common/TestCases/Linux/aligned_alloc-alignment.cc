// RUN: %clangxx %collect_stack_traces -O0 %s -o %t

// Alignment is not a power of 2:
// RUN: %env_tool_opts=allocator_may_return_null=0 not %run %t 17 2>&1 | FileCheck %s
// Size is not a multiple of alignment:
// RUN: %env_tool_opts=allocator_may_return_null=0 not %run %t 8 2>&1 | FileCheck %s
// Alignment is 0:
// RUN: %env_tool_opts=allocator_may_return_null=0 not %run %t 0 2>&1 | FileCheck %s

// The same for allocator_may_return_null=1:
// RUN: %env_tool_opts=allocator_may_return_null=1 %run %t 17 2>&1 | FileCheck %s --check-prefix=CHECK-NULL
// RUN: %env_tool_opts=allocator_may_return_null=1 %run %t 8 2>&1 | FileCheck %s --check-prefix=CHECK-NULL
// RUN: %env_tool_opts=allocator_may_return_null=1 %run %t 0 2>&1 | FileCheck %s --check-prefix=CHECK-NULL

// REQUIRES: stable-runtime

// UNSUPPORTED: android, ubsan

#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

extern void *aligned_alloc(size_t alignment, size_t size);

int main(int argc, char **argv) {
  assert(argc == 2);
  const int alignment = atoi(argv[1]);

  void *p = aligned_alloc(alignment, 100);
  // CHECK: {{ERROR: .*Sanitizer: invalid alignment requested in aligned_alloc}}
  // Handle a case when aligned_alloc is aliased by memalign.
  // CHECK: {{#0 .*}}{{aligned_alloc|memalign}}
  // CHECK: {{#1 .*main .*aligned_alloc-alignment.cc:}}[[@LINE-4]]
  // CHECK: {{SUMMARY: .*Sanitizer: invalid-aligned-alloc-alignment}}

  // The NULL pointer is printed differently on different systems, while (long)0
  // is always the same.
  fprintf(stderr, "errno: %d, p: %lx\n", errno, (long)p);
  // CHECK-NULL: errno: 22, p: 0

  return 0;
}
