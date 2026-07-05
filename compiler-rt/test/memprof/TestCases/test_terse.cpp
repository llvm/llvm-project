// Check terse format profile with a single malloc call and set of loads and
// stores. Ensures we get the same profile regardless of whether the memory is
// deallocated before exit.

// RUN: %clangxx_memprof -O0 %s -o %t
// RUN: %env_memprof_opts=print_text=true:log_path=stderr:print_terse=1 %run %t 2>&1 | FileCheck %s

// RUN: %clangxx_memprof -DFREE -O0 %s -o %t
// RUN: %env_memprof_opts=print_text=true:log_path=stderr:print_terse=1 %run %t 2>&1 | FileCheck %s

// CHECK: MIB:[[STACKID:[0-9]+]]/1/40.00/40/40/20.00/20/20/[[AVELIFETIME:[0-9]+]].00/[[AVELIFETIME]]/[[AVELIFETIME]]/{{[01]}}/0/0/0
// CHECK: Stack for id [[STACKID]]:
// CHECK-NEXT: #0 {{.*}} in operator new
// CHECK-NEXT: #1 {{.*}} in main {{.*}}:[[@LINE+6]]

#include <stdio.h>
#include <stdlib.h>

int main() {
  int *p = new int[10];
  for (int i = 0; i < 10; i++)
    p[i] = i;
  int j = 0;
  for (int i = 0; i < 10; i++)
    j += p[i];
#ifdef FREE
  delete[] p;
#endif

  return 0;
}
