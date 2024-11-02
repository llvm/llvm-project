// Print matched suppressions only if print_suppressions=1 AND at least one is
// matched. Default is print_suppressions=true.
// RUN: %clangxx_lsan %s -o %t
// RUN: %env_lsan_opts=use_registers=0:use_stacks=0:print_suppressions=0 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-dont-print
// RUN: %env_lsan_opts=use_registers=0:use_stacks=0 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-dont-print
// RUN: %env_lsan_opts=use_registers=0:use_stacks=0:print_suppressions=0 %run %t foo 2>&1 | FileCheck %s --check-prefix=CHECK-dont-print
// RUN: %env_lsan_opts=use_registers=0:use_stacks=0 %run %t foo 2>&1 | FileCheck %s --check-prefix=CHECK-print

#include <stdio.h>
#include <stdlib.h>

#include "sanitizer/lsan_interface.h"

extern "C"
const char *__lsan_default_suppressions() {
  return "leak:*LSanTestLeakingFunc*";
}

void LSanTestLeakingFunc() {
  void *p = malloc(666);
  fprintf(stderr, "Test alloc: %p.\n", p);
}

int main(int argc, char **argv) {
  printf("print for nonempty output\n");
  if (argc > 1)
    LSanTestLeakingFunc();
  return 0;
}
// CHECK-print: Suppressions used:
// CHECK-print: 1 666 *LSanTestLeakingFunc*
// CHECK-dont-print-NOT: Suppressions used:
