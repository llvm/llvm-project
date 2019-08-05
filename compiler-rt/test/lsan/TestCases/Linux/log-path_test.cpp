// RUN: %clangxx_lsan %s -o %t
// The globs below do not work in the lit shell.

// Regular run.
// RUN: %env_lsan_opts="use_stacks=0" not %run %t > %t.out 2>&1
// RUN: FileCheck %s --check-prefix=CHECK-ERROR < %t.out

// Good log_path.
// RUN: rm -f %t.log.*
// RUN: %env_lsan_opts="use_stacks=0:log_path='"%t.log"'" not %run %t > %t.out 2>&1
// RUN: FileCheck %s --check-prefix=CHECK-ERROR < %t.log.*

#include <stdio.h>
#include <stdlib.h>
#include "sanitizer_common/print_address.h"

int main() {
  void *stack_var = malloc(1337);
  print_address("Test alloc: ", 1, stack_var);
  // Do not return from main to prevent the pointer from going out of scope.
  exit(0);
}

// CHECK-ERROR: LeakSanitizer: detected memory leaks
// CHECK-ERROR: Direct leak of 1337 byte(s) in 1 object(s) allocated from
// CHECK-ERROR: SUMMARY: {{(Leak|Address)}}Sanitizer:
