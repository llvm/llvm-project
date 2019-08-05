// RUN: %clangxx_asan -O0 %s -o %t
// RUN: %env_asan_opts=allocator_may_return_null=0 not %run %t 2>&1 | FileCheck %s
// RUN: %env_asan_opts=allocator_may_return_null=1 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-NULL

// REQUIRES: stable-runtime

#include <stdio.h>
#include <stdlib.h>

static const size_t kMaxAllowedMallocSizePlusOne =
#if __LP64__ || defined(_WIN64)
    (1ULL << 40) + 1;
#else
    (3UL << 30) + 1;
#endif

int main() {
  void *p = malloc(kMaxAllowedMallocSizePlusOne);
  // CHECK: {{ERROR: AddressSanitizer: requested allocation size .* \(.* after adjustments for alignment, red zones etc\.\) exceeds maximum supported size}}
  // CHECK: {{#0 0x.* in .*malloc}}
  // CHECK: {{#1 0x.* in main .*malloc-size-too-big.cpp:}}[[@LINE-3]]
  // CHECK: SUMMARY: AddressSanitizer: allocation-size-too-big

  printf("malloc returned: %zu\n", (size_t)p);
  // CHECK-NULL: malloc returned: 0

  return 0;
}
