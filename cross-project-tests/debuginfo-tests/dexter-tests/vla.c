// This test case verifies the debug location for variable-length arrays.
// REQUIRES: lldb
// UNSUPPORTED: system-windows
//
// RUN: %clang -std=gnu11 -O0 -glldb %s -o %t
// RUN: %dexter -w --use-script --binary %t %dexter_lldb_args -- %s \
// RUN:  | FileCheck %s

void init_vla(int size) {
  int i;
  int vla[size];
  for (i = 0; i < size; i++)
    vla[i] = size-i;
  vla[0] = size; // !dex_label end_init
}

int main(int argc, const char **argv) {
  init_vla(23);
  return 0;
}

// CHECK-DAG: seen_values: 2
// CHECK-DAG: correct_step_coverage: 100.0%

/*
---
!where {lines: !label end_init}:
  !value vla:
    "[0]": 23
    "[1]": 22
...
*/
