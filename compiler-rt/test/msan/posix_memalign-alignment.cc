// RUN: %clangxx_msan -fsanitize-memory-track-origins -O0 -g %s -o %t
// RUN: MSAN_OPTIONS=$MSAN_OPTIONS:allocator_may_return_null=0 not %run %t 2>&1 | FileCheck %s
// RUN: MSAN_OPTIONS=$MSAN_OPTIONS:allocator_may_return_null=1 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-NULL

// REQUIRES: stable-runtime

#include <stdio.h>
#include <stdlib.h>

int main() {
  void *p = reinterpret_cast<void*>(42);
  int res = posix_memalign(&p, 17, 100);
  // CHECK: ERROR: MemorySanitizer: invalid alignment requested in posix_memalign: 17
  // Check just the top frame since mips is forced to use store_context_size==1
  // CHECK: {{#0 0x.* in .*posix_memalign}}
  // CHECK: SUMMARY: MemorySanitizer: invalid-posix-memalign-alignment

  printf("pointer after failed posix_memalign: %zd\n", (size_t)p);
  // CHECK-NULL: pointer after failed posix_memalign: 42

  return 0;
}
