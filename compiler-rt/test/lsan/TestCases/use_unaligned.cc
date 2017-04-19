// Test that unaligned pointers are detected correctly.
// RUN: LSAN_BASE="detect_leaks=1:report_objects=1:use_stacks=0:use_registers=0"
// RUN: %clangxx_lsan %s -o %t
// RUN: %env_lsan_opts=$LSAN_BASE:"use_unaligned=0" not %run %t 2>&1 | FileCheck %s
// RUN: %env_lsan_opts=$LSAN_BASE:"use_unaligned=1" %run %t 2>&1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "sanitizer_common/print_address.h"

void *arr[2];

int main() {
  void *p = malloc(1337);
  print_address("Test alloc: ", 1, p);
  char *char_arr = (char *)arr;
  memcpy(char_arr + 1, &p, sizeof(p));
  return 0;
}
// CHECK: Test alloc: [[ADDR:0x[0-9,a-f]+]]
// CHECK: LeakSanitizer: detected memory leaks
// CHECK: [[ADDR]] (1337 bytes)
// CHECK: SUMMARY: {{(Leak|Address)}}Sanitizer:
