// Check that merging of MIB info (min/max size and access counts specifically)
// is done correctly.

// RUN: %clangxx_memprof -O0 %s -o %t
// RUN: %env_memprof_opts=print_text=true:log_path=stderr %run %t 2>&1 | FileCheck %s

// This is actually:
//  Memory allocation stack id = STACKID
//   alloc_count 2, size (ave/min/max) 60.00 / 40 / 80
// but we need to look for them in the same CHECK to get the correct STACKID.
// CHECK:  Memory allocation stack id = [[STACKID:[0-9]+]]{{[[:space:]].*}}alloc_count 2, size (ave/min/max) 60.00 / 40 / 80
// CHECK-NEXT:  access_count (ave/min/max): 30.00 / 20 / 40
// Unfortunately there is not a reliable way to check the ave/min/max lifetime.
// CHECK-NEXT:  lifetime (ave/min/max):
// CHECK-NEXT:  num migrated: {{[0-1]}}, num lifetime overlaps: 0, num same alloc cpu: 1, num same dealloc_cpu: 1
// CHECK: Stack for id [[STACKID]]:
// CHECK-NEXT: #0 {{.*}} in operator new
// CHECK-NEXT: #1 {{.*}} in main {{.*}}:[[@LINE+7]]

#include <stdio.h>
#include <stdlib.h>

int main() {
  for (int j = 1; j < 3; j++) {
    int *p = new int[10 * j];
    for (int i = 0; i < 10 * j; i++)
      p[i] = i;
    int a = 0;
    for (int i = 0; i < 10 * j; i++)
      a += p[i];
    delete[] p;
  }

  return 0;
}
