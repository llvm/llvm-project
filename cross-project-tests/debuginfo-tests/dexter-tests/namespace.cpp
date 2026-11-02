// Purpose:
// Ensure that the debug information for a global variable includes
// namespace information.

// REQUIRES: lldb
// UNSUPPORTED: system-windows

// RUN: %clang++ -g -O0 %s -o %t
// RUN: %dexter -w \
// RUN:     --binary %t %dexter_lldb_args -v -- %s | FileCheck %s

#include <stdio.h>

namespace monkey {
const int ape = 32;
}

int main() {
  printf("hello %d\n", monkey::ape); // !dex_label main
  return 0;
}

// CHECK-DAG: seen_values: 1
// CHECK-DAG: correct_step_coverage: 100.0%

/*
---
!where {lines: !label main}:
  !value "monkey::ape": 32
...
*/
