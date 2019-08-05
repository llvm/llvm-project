// Test for incorrect use of __lsan_ignore_object().
// RUN: %clangxx_lsan %s -o %t
// RUN: %env_lsan_opts=$LSAN_BASE %run %t 2>&1 | FileCheck %s

#include <stdio.h>
#include <stdlib.h>

#include "sanitizer/lsan_interface.h"

int main() {
  void *p = malloc(1337);
  fprintf(stderr, "Test alloc: %p.\n", p);
  __lsan_ignore_object(p);
  __lsan_ignore_object(p);
  free(p);
  __lsan_ignore_object(p);
  return 0;
}
// CHECK: Test alloc: [[ADDR:.*]].
// CHECK-NOT: SUMMARY: {{.*}} leaked
