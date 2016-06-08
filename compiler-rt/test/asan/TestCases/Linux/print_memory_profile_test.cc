// Printing memory profiling only works in the configuration where we can
// detect leaks.
// REQUIRES: leak-detection
//
// RUN: %clangxx_asan %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s
#include <sanitizer/common_interface_defs.h>

#include <stdio.h>

char *sink[1000];

int main() {
  int idx = 0;
  for (int i = 0; i < 17; i++)
    sink[idx++] = new char[131000];
  for (int i = 0; i < 28; i++)
    sink[idx++] = new char[24000];

  __sanitizer_print_memory_profile(100);
  __sanitizer_print_memory_profile(50);
}

// CHECK: Live Heap Allocations: {{.*}}; showing top 100%
// CHECK: 2227000 byte(s) ({{.*}}%) in 17 allocation(s)
// CHECK: 672000 byte(s) ({{.*}}%) in 28 allocation(s)
// CHECK: Live Heap Allocations: {{.*}}; showing top 50%
// CHECK: 2227000 byte(s) ({{.*}}%) in 17 allocation(s)
// CHECK-NOT: 1008 byte
