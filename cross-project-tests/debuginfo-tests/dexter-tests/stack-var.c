// REQUIRES: lldb
// UNSUPPORTED: system-windows
//
// RUN: %clang -std=gnu11 -O -glldb %s -o %t
// RUN: %dexter -w --use-script --binary %t %dexter_lldb_args -- %s \
// RUN:  | FileCheck %s

void __attribute__((noinline, optnone)) bar(int *test) {}
int main() {
  int test;
  test = 23;
  bar(&test);  // !dex_label before_bar
  return test; // !dex_label after_bar
}

// CHECK-DAG: seen_values: 2
// CHECK-DAG: correct_step_coverage: 100.0%

/*
---
!where {lines: !label before_bar}:
  !value test: 23
!where {lines: !label after_bar}:
  !value test: 23
...
*/
