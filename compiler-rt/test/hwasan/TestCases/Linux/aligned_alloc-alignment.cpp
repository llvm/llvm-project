// RUN: %clangxx_hwasan -O0 %s -o %t
// RUN: %env_hwasan_opts=allocator_may_return_null=0 not %run %t 2>&1 | FileCheck %s
// RUN: %env_hwasan_opts=allocator_may_return_null=1 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-NULL

// UNSUPPORTED: android

// REQUIRES: stable-runtime

#include <stdio.h>
#include <stdlib.h>

#include "../utils.h"

extern void *aligned_alloc(size_t alignment, size_t size);

int main() {
  void *p = aligned_alloc(17, 100);
  // CHECK: ERROR: HWAddressSanitizer: invalid alignment requested in aligned_alloc: 17
  // CHECK: {{#0 0x.* in .*}}{{aligned_alloc|memalign}}
  // CHECK: {{#[12] 0x.* in main .*aligned_alloc-alignment.cpp:}}[[@LINE-3]]
  // CHECK: SUMMARY: HWAddressSanitizer: invalid-aligned-alloc-alignment

  untag_printf("pointer after failed aligned_alloc: %zd\n", (size_t)p);
  // CHECK-NULL: pointer after failed aligned_alloc: 0

  return 0;
}
