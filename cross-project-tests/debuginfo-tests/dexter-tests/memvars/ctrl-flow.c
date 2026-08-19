// REQUIRES: lldb
// UNSUPPORTED: system-windows
// RUN: %clang -std=gnu11 -O2 -glldb %s -o %t
// RUN: %dexter -w %dexter_lldb_args --binary %t -- %s | FileCheck %s

//// Check that we give good locations to a variable ('local') which is escaped
//// down some control paths and not others. This example is handled well currently.

int g;
__attribute__((__noinline__))
void leak(int *ptr) {
  g = *ptr;
  *ptr = 2;
}

__attribute__((__noinline__))
int fun(int cond) {
  int local = 0; // !dex_label s1
  if (cond)
    leak(&local);
  else
    local = 1;
  return local; // !dex_label s2
}

int main() {
  int a = fun(1);
  int b = fun(0);
  return a + b;
}

// CHECK-DAG: seen_values: 3
// CHECK-DAG: correct_step_coverage: 100.0%

/*
---
!where {function: "fun"}:
  !and {lines: !label s1}:
    !value local: 0
  !and {lines: !label s2}:
    !value local: [2, 1]
...
*/
