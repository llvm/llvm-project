// XFAIL:*
//// See PR47946.

// REQUIRES: lldb
// UNSUPPORTED: system-windows
// RUN: %clang -std=gnu11 -O2 -glldb %s -o %t
// RUN: %dexter -w --use-script %dexter_lldb_args --binary %t -- %s \
// RUN:   | FileCheck %s
//
//// Check that once-escaped variable 'param' can still be read after we
//// perform inlining + mem2reg, and that we see the DSE'd value 255.


int g;
__attribute__((__always_inline__))
static void use(int* p) {
  g = *p;
  *p = 255;
  volatile int step = 0;  // !dex_label use1
}

__attribute__((__noinline__))
void fun(int param) {
  //// Make sure first step is in 'fun'.
  volatile int step = 0;  // !dex_label fun1
  use(&param);
  return;                 // !dex_label fun2
}

int main() {
  fun(5);
}

// CHECK-DAG: seen_values: 3
// CHECK-DAG: correct_step_coverage: 100.0%

/*
---
!where {function: fun}:
  # Expect param == 5 before stepping through inlined 'use'.
  !and {lines: !label fun1}:
    !value param: 5
  # Expect param == 255 after assignment in inlined frame 'use'.
  !and {lines: !label fun2}:
    !value param: 255
  !where {file: "inlining-dse.c", lines: !label use1}:
    !and {at_frame_idx: 1}:
      !value param: 255
...
*/
