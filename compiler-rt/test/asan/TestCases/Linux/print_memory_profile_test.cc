// RUN: %clangxx_asan %s -o %t
// RUN: %t 2>&1 | FileCheck %s
#include <sanitizer/common_interface_defs.h>

#include <stdio.h>

char *sink[1000];

int main() {
  int idx = 0;
  for (int i = 0; i < 17; i++)
    sink[idx++] = new char[131];
  for (int i = 0; i < 42; i++)
    sink[idx++] = new char[24];

  __sanitizer_print_memory_profile(100);
  __sanitizer_print_memory_profile(50);
}

// CHECK: Live Heap Allocations: {{.*}}; showing top 100%
// CHECK: 2227 byte(s) ({{.*}}%) in 17 allocation(s)
// CHECK: 1008 byte(s) ({{.*}}%) in 42 allocation(s)
// CHECK: Live Heap Allocations: {{.*}}; showing top 50%
// CHECK: 2227 byte(s) ({{.*}}%) in 17 allocation(s)
// CHECK-NOT: 1008 byte
