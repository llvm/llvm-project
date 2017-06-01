// Printing memory profiling only works in the configuration where we can
// detect leaks.
// REQUIRES: leak-detection
//
// RUN: %clangxx_asan %s -o %t
// RUN: %run %t 100 2>&1 | FileCheck %s --check-prefix=CHECK-100
// RUN: %run %t 50 2>&1 | FileCheck %s --check-prefix=CHECK-50
#include <sanitizer/common_interface_defs.h>

#include <stdio.h>
#include <stdlib.h>

char *sink[1000];

int main(int argc, char **argv) {
  if (argc < 2)
    return 1;

  int idx = 0;
  for (int i = 0; i < 17; i++)
    sink[idx++] = new char[131000];
  for (int i = 0; i < 28; i++)
    sink[idx++] = new char[24000];

  __sanitizer_print_memory_profile(atoi(argv[1]));
}

// CHECK-100: Live Heap Allocations: {{.*}}; showing top 100%
// CHECK-100: 2227000 byte(s) ({{.*}}%) in 17 allocation(s)
// CHECK-100: 672000 byte(s) ({{.*}}%) in 28 allocation(s)
// CHECK-50: Live Heap Allocations: {{.*}}; showing top 50%
// CHECK-50: 2227000 byte(s) ({{.*}}%) in 17 allocation(s)
// CHECK-50-NOT: allocation
