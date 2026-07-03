// Purpose:
// Ensure that debug information for a local variable does not hide
// a global definition that has the same name.

// REQUIRES: lldb
// UNSUPPORTED: system-windows
// RUN: %clang++ -std=gnu++11 -O0 -g %s -o %t
// RUN: %dexter -w --use-script \
// RUN:     --binary  %t %dexter_lldb_args -v -- %s | FileCheck %s

const int d = 100;

extern int foo();

int main() {
  const int d = 4;
  const float e = 4; // !dex_label main
  const char *f = "Woopy";
  return d + foo();
}

int foo() {
  return d; // !dex_label foo
}

// CHECK-DAG: seen_values: 2
// CHECK-DAG: correct_step_coverage: 100.0%

/*
---
!where {lines: !label main}:
  !value d: 4
!where {lines: !label foo}:
  !value d: 100
...
*/
