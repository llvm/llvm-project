// RUN: %clangxx -O0 %s -o %t
// RUN: %tool_options=allocator_may_return_null=0 not %run %t 17 2>&1 | FileCheck %s
// RUN: %tool_options=allocator_may_return_null=0 not %run %t 0 2>&1 | FileCheck %s
// RUN: %tool_options=allocator_may_return_null=1 %run %t 17 2>&1 | FileCheck %s --check-prefix=CHECK-NULL
// RUN: %tool_options=allocator_may_return_null=1 %run %t 0 2>&1 | FileCheck %s --check-prefix=CHECK-NULL

// UNSUPPORTED: android, msan, tsan, ubsan

// REQUIRES: stable-runtime

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

extern void *aligned_alloc(size_t alignment, size_t size);

int main(int argc, char **argv) {
  assert(argc == 2);
  const int alignment = atoi(argv[1]);

  void *p = aligned_alloc(alignment, 100);
  // CHECK: {{ERROR: .*Sanitizer: invalid alignment requested in aligned_alloc}}
  // Handle a case when aligned_alloc is aliased by memalign.
  // CHECK: {{#0 0x.* in .*}}{{aligned_alloc|memalign}}
  // CHECK: {{#1 0x.* in main .*aligned_alloc-alignment.cc:}}[[@LINE-4]]
  // CHECK: {{SUMMARY: .*Sanitizer: invalid-aligned-alloc-alignment}}

  printf("pointer after failed aligned_alloc: %zd\n", (size_t)p);
  // CHECK-NULL: pointer after failed aligned_alloc: 0

  return 0;
}
