// Test for ScopedDisabler.
// RUN: %clangxx_lsan %s -o %t
// RUN: %env_lsan_opts=report_objects=1:use_registers=0:use_stacks=0:use_tls=0 not %run %t 2>&1 | FileCheck %s

#include <stdio.h>
#include <stdlib.h>

#include "sanitizer/lsan_interface.h"

int main() {
  void **p;
  {
    __lsan::ScopedDisabler d;
    p = new void *;
    fprintf(stderr, "Test alloc p: %p.\n", p);
  }
  *p = malloc(666);
  void *q = malloc(1337);
  fprintf(stderr, "Test alloc q: %p.\n", q);
  return 0;
}

// CHECK: Test alloc p: [[ADDR:.*]].
// CHECK-NOT: [[ADDR]]
