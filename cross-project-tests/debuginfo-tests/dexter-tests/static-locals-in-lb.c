// This test case verifies that function-local static variables with the same name
// declared in different lexical blocks are read correctly in the debugger.
// REQUIRES: lldb
// UNSUPPORTED: system-windows
//
// RUN: %clang -O0 -glldb %s -o %t
// RUN: %dexter -w --binary %t %dexter_lldb_args -- %s | FileCheck %s

int test(int x) {
  int result;
  if (x > 0) {
    static int y = 1;
    result = x + y; // !dex_label pos
  } else {
    static int y = 2;
    result = x - y; // !dex_label neg
  }
  return result; // !dex_label ret
}

int main(int argc, const char **argv) {
  test(5);
  test(-3);
  return 0;
}

// CHECK-DAG: seen_values: 2
// CHECK-DAG: correct_steps: 2
// CHECK-DAG: unexpected_value_steps: 0
// CHECK-DAG: missing_var_steps: 2
// CHECK-DAG: correct_step_coverage: 50.0%

/*
---
!where {lines: !label pos}:
  !value y: 1
!where {lines: !label neg}:
  !value y: 2
# The value of y's should not be available out of declaration scopes.
!where {lines: !label ret}:
  !value y: null
...
*/
