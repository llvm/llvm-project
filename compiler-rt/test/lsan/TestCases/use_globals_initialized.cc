// Test that initialized globals are included in the root set.
// RUN: LSAN_BASE="report_objects=1:use_stacks=0:use_registers=0"
// RUN: %clangxx_lsan %s -o %t
// RUN: %env_lsan_opts=$LSAN_BASE:"use_globals=0" not %run %t 2>&1 | FileCheck %s
// RUN: %env_lsan_opts=$LSAN_BASE:"use_globals=1" %run %t 2>&1
// RUN: %env_lsan_opts="" %run %t 2>&1

#include <stdio.h>
#include <stdlib.h>
#include "sanitizer_common/print_address.h"

void *data_var = (void *)1;

int main() {
  data_var = malloc(1337);
  print_address("Test alloc: ", 1, data_var);
  return 0;
}
// CHECK: Test alloc: [[ADDR:0x[0-9,a-f]+]]
// CHECK: LeakSanitizer: detected memory leaks
// CHECK: [[ADDR]] (1337 bytes)
// CHECK: SUMMARY: {{(Leak|Address)}}Sanitizer:
