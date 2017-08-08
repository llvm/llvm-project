// ASan-poisoned memory should be ignored if use_poisoned is false.
// REQUIRES: asan
// RUN: LSAN_BASE="report_objects=1:use_stacks=0:use_registers=0"
// RUN: %clangxx_lsan %s -o %t
// RUN: %env_lsan_opts=$LSAN_BASE:"use_poisoned=0" not %run %t 2>&1 | FileCheck %s
// RUN: %env_lsan_opts=$LSAN_BASE:"use_poisoned=1" %run %t 2>&1

#include <stdio.h>
#include <stdlib.h>
#include <sanitizer/asan_interface.h>
#include <assert.h>
#include "sanitizer_common/print_address.h"

void **p;

int main() {
  p = new void *;
  *p = malloc(1337);
  print_address("Test alloc: ", 1, *p);
  __asan_poison_memory_region(p, sizeof(*p));
  return 0;
}
// CHECK: Test alloc: [[ADDR:0x[0-9,a-f]+]]
// CHECK: LeakSanitizer: detected memory leaks
// CHECK: [[ADDR]] (1337 bytes)
// CHECK: SUMMARY: AddressSanitizer:
