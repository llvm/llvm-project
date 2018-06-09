// RUN: %clangxx_msan -fsanitize-memory-track-origins -O0 -g %s -o %t
// RUN: MSAN_OPTIONS=allocator_may_return_null=0 not %run %t 2>&1 | FileCheck %s
// RUN: MSAN_OPTIONS=allocator_may_return_null=1 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-NULL

// UNSUPPORTED: android

// REQUIRES: stable-runtime

#include <stdio.h>
#include <stdlib.h>

extern void *aligned_alloc(size_t alignment, size_t size);

int main() {
  void *p = aligned_alloc(17, 100);
  // CHECK: ERROR: MemorySanitizer: invalid alignment requested in aligned_alloc: 17
  // Check just the top frame since mips is forced to use store_context_size==1
  // and also handle a case when aligned_alloc is aliased by memalign.
  // CHECK: {{#0 0x.* in .*}}{{aligned_alloc|memalign}}
  // CHECK: SUMMARY: MemorySanitizer: invalid-aligned-alloc-alignment

  printf("pointer after failed aligned_alloc: %zd\n", (size_t)p);
  // CHECK-NULL: pointer after failed aligned_alloc: 0

  return 0;
}
