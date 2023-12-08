// Test that fake stack (introduced by ASan's use-after-return mode) is included
// in the root set.
// RUN: %clangxx_lsan %s -O2 -o %t
// RUN: env ASAN_OPTIONS=detect_stack_use_after_return=1 %env_lsan_opts="report_objects=1:use_registers=0:use_stacks=0" not %run %t 2>&1 | FileCheck %s
// RUN: env ASAN_OPTIONS=detect_stack_use_after_return=1 %env_lsan_opts="report_objects=1:use_registers=0:use_stacks=1" %run %t 2>&1
// RUN: env ASAN_OPTIONS=detect_stack_use_after_return=1 %env_lsan_opts="" %run %t 2>&1

// Investigate why it does not fail with use_stack=0
// UNSUPPORTED: arm-linux || armhf-linux

#include <stdio.h>
#include <stdlib.h>
#include "sanitizer_common/print_address.h"

int main() {
  void *stack_var = malloc(1337);
  print_address("Test alloc: ", 1, stack_var);
  // Take pointer to variable, to ensure it's not optimized into a register.
  print_address("Stack var at: ", 1, &stack_var);
  // Do not return from main to prevent the pointer from going out of scope.
  exit(0);
}
// CHECK: Test alloc: [[ADDR:0x[0-9,a-f]+]]
// CHECK: LeakSanitizer: detected memory leaks
// CHECK: [[ADDR]] (1337 bytes)
// CHECK: SUMMARY: {{.*}}Sanitizer:
