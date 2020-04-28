// RUN: %clangxx_hwasan -O0 %s -o %t
// RUN: %env_hwasan_opts=allocator_may_return_null=0 not %run %t 2>&1 | FileCheck %s
// RUN: %env_hwasan_opts=allocator_may_return_null=1 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-NULL

// REQUIRES: stable-runtime

#include <stdio.h>
#include <stdlib.h>

#include "../utils.h"

int main() {
  void *p = reinterpret_cast<void*>(42);
  int res = posix_memalign(&p, 17, 100);
  // CHECK: ERROR: HWAddressSanitizer: invalid alignment requested in posix_memalign: 17
  // CHECK: {{#0 0x.* in .*posix_memalign}}
  // CHECK: {{#[12] 0x.* in main .*posix_memalign-alignment.cpp:}}[[@LINE-3]]
  // CHECK: SUMMARY: HWAddressSanitizer: invalid-posix-memalign-alignment

  untag_printf("pointer after failed posix_memalign: %zd\n", (size_t)p);
  // CHECK-NULL: pointer after failed posix_memalign: 42

  return 0;
}
