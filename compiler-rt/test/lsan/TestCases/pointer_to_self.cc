// Regression test: pointers to self should not confuse LSan into thinking the
// object is indirectly leaked. Only external pointers count.
// RUN: LSAN_BASE="detect_leaks=1:report_objects=1:use_registers=0"
// RUN: %clangxx_lsan %s -o %t
// RUN: %env_lsan_opts=$LSAN_BASE:"use_stacks=0" not %run %t 2>&1 | FileCheck %s

#include <stdio.h>
#include <stdlib.h>
#include "sanitizer_common/print_address.h"

int main() {
  void *p = malloc(1337);
  *reinterpret_cast<void **>(p) = p;
  print_address("Test alloc: ", 1, p);
}
// CHECK: Test alloc: [[ADDR:0x[0-9,a-f]+]]
// CHECK: LeakSanitizer: detected memory leaks
// CHECK: [[ADDR]] (1337 bytes)
// CHECK: SUMMARY: {{(Leak|Address)}}Sanitizer:
