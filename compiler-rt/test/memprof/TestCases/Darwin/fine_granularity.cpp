// Test that -memprof-fine-granularity works end-to-end on Darwin.
// On Apple platforms, the runtime uses 8-byte shadow granularity with u8
// counters. The -memprof-fine-granularity flag makes the instrumentation pass
// emit 8-byte granularity accesses with i8 shadow type, matching the runtime.
// This ensures access_count is correctly reported (non-zero).

// RUN: %clangxx_memprof -O0 -mllvm -memprof-fine-granularity -mllvm -memprof-use-callbacks=true %s -o %t
// RUN: %env_memprof_opts=print_text=true:log_path=stderr %run %t 2>&1 | FileCheck %s

#include <cstdlib>

int main() {
  // Use a distinctive size (100) to avoid matching system allocations.
  char *buf = (char *)malloc(100);
  // Perform several accesses to ensure non-zero access_count.
  for (int i = 0; i < 10; i++)
    buf[0] = 'A';
  for (int i = 0; i < 5; i++)
    buf[8] = 'B';
  free(buf);
  return 0;
}

// CHECK: alloc_count 1, size (ave/min/max) 100.00 / 100 / 100
// CHECK-NEXT: access_count (ave/min/max): {{[1-9][0-9]*}}
