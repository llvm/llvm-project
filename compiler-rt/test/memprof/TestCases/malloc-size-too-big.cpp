// RUN: %clangxx_memprof -O0 %s -o %t
// RUN: %env_memprof_opts=log_path=stderr:allocator_may_return_null=0 not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-SUMMARY
// RUN: %env_memprof_opts=log_path=stderr:allocator_may_return_null=1 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-NULL
// Test print_summary
// RUN: %env_memprof_opts=log_path=stderr:allocator_may_return_null=0:print_summary=0 not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-NOSUMMARY

#include <stdio.h>
#include <stdlib.h>

static const size_t kMaxAllowedMallocSizePlusOne = (1ULL << 40) + 1;
int main() {
  void *p = malloc(kMaxAllowedMallocSizePlusOne);
  // CHECK: {{ERROR: MemProfiler: requested allocation size .* exceeds maximum supported size}}
  // CHECK: {{#0 0x.* in .*malloc}}
  // CHECK: {{#1 0x.* in main .*malloc-size-too-big.cpp:}}[[@LINE-3]]
  // CHECK-SUMMARY: SUMMARY: MemProfiler: allocation-size-too-big
  // CHECK-NOSUMMARY-NOT: SUMMARY:

  printf("malloc returned: %zu\n", (size_t)p);
  // CHECK-NULL: malloc returned: 0

  return 0;
}
