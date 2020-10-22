// Test for __lsan_ignore_object().
// RUN: LSAN_BASE="report_objects=1:use_registers=0:use_stacks=0:use_globals=0:use_tls=0"
// RUN: %clang_lsan %s -o %t
// RUN: %env_lsan_opts=$LSAN_BASE not %run %t 2>&1 | FileCheck %s

#include <stdio.h>
#include <stdlib.h>

#include "sanitizer/lsan_interface.h"

int main() {
  // Explicitly ignored object.
  void **p = malloc(sizeof(void *));
  // Transitively ignored object.
  *p = malloc(666);
  // Non-ignored object.
  volatile void *q = malloc(1337);
  fprintf(stderr, "Test alloc: %p.\n", p);
  __lsan_ignore_object(p);
  return 0;
}
// CHECK: Test alloc: [[ADDR:.*]].
// CHECK: SUMMARY: {{(Leak|Address)}}Sanitizer: 1337 byte(s) leaked in 1 allocation(s)
