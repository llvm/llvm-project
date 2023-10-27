// Test that statically allocated TLS space is included in the root set.
// RUN: %clangxx_lsan %s -o %t
// RUN: %env_lsan_opts="report_objects=1:use_stacks=0:use_registers=0:use_tls=0" not %run %t 2>&1 | FileCheck %s
// RUN: %env_lsan_opts="report_objects=1:use_stacks=0:use_registers=0:use_tls=1" %run %t 2>&1
// RUN: %env_lsan_opts="" %run %t 2>&1

#include <stdio.h>
#include <stdlib.h>
#include "sanitizer_common/print_address.h"

__thread void *tls_var;

int main() {
  tls_var = malloc(1337);
  print_address("Test alloc: ", 1, tls_var);
  return 0;
}
// CHECK: Test alloc: [[ADDR:0x[0-9,a-f]+]]
// CHECK: LeakSanitizer: detected memory leaks
// CHECK: [[ADDR]] (1337 bytes)
// CHECK: SUMMARY: {{.*}}Sanitizer:
