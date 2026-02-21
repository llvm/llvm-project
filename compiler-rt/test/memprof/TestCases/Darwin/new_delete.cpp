// Test that memprof works with operator new/delete on Darwin.
// This verifies C++ allocation interception through DYLD interposition.

// RUN: %clangxx_memprof -O0 %s -o %t
// RUN: %env_memprof_opts=print_text=true:log_path=stderr:print_module_map=1 %run %t 2>%t.out
// RUN: FileCheck %s < %t.out
// Verify that raw addresses resolve to the expected symbol.
// RUN: %python %S/symbolize_raw_stacks.py %t %t.out | FileCheck --check-prefix=SYM %s

// CHECK:  Memory allocation stack id = [[STACKID:[0-9]+]]{{[[:space:]].*}}alloc_count 1, size (ave/min/max) 40.00 / 40 / 40
// CHECK-NEXT:  access_count (ave/min/max): 20.00 / 20 / 20
// CHECK: Stack for id [[STACKID]]:
// CHECK-NEXT: #0 0x{{[0-9a-f]+}}
// CHECK-NEXT: #1 0x{{[0-9a-f]+}}

// SYM: main

#include <cstdlib>

int main() {
  int *p = new int[10];
  for (int i = 0; i < 10; i++)
    p[i] = i;
  int j = 0;
  for (int i = 0; i < 10; i++)
    j += p[i];
  delete[] p;
  return 0;
}
