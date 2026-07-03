// REQUIRES: lldb
// UNSUPPORTED: system-windows
// RUN: %clang -std=gnu11 -O2 -glldb %s -o %t
// RUN: %dexter -w --use-script %dexter_lldb_args --binary %t -- %s \
// RUN:   | FileCheck %s
//
//// Check that the once-escaped variable 'param' can still be read after
//// we perform inlining + mem2reg. See D89810 and D85555.

int g;
__attribute__((__always_inline__))
static void use(int* p) {
  g = *p;
}

__attribute__((__noinline__))
void fun(int param) {
  volatile int step1 = 0; // !dex_label s1
  use(&param);
  volatile int step2 = 0; // !dex_label s2
}

int main() {
  fun(5);
}

// CHECK-DAG: seen_values: 1
// CHECK-DAG: correct_step_coverage: 100.0%

/*
---
!where {lines: !range [!label s1, !label s2]}:
  !value param: 5
...
*/
