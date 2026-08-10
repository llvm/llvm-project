// Test for __lsan_disable() / __lsan_enable().
// RUN: %clang_lsan %s -o %t
// RUN: %env_lsan_opts=report_objects=1:use_registers=0:use_stacks=0 not %run %t 2>&1 | FileCheck %s

// Investigate why it does not fail with use_tls=0
// UNSUPPORTED: arm-linux || armhf-linux

#include <stdio.h>
#include <stdlib.h>

#include "sanitizer/lsan_interface.h"

int main() {
  void **p;
  {
    __lsan_disable();
    p = malloc(sizeof(void *));
    __lsan_enable();
  }
  *p = malloc(666);
  void *q = malloc(1337);
  // Break optimization.
  fprintf(stderr, "Test alloc: %p.\n", q);
  return 0;
}
// CHECK: SUMMARY: {{.*}}Sanitizer: 1337 byte(s) leaked in 1 allocation(s)
