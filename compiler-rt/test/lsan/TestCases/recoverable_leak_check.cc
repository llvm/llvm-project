// Test for on-demand leak checking.
// RUN: LSAN_BASE="detect_leaks=1:use_stacks=0:use_registers=0"
// RUN: %clangxx_lsan %s -o %t
// RUN: LSAN_OPTIONS=$LSAN_BASE %run %t foo 2>&1 | FileCheck %s
// RUN: LSAN_OPTIONS=$LSAN_BASE %run %t 2>&1 | FileCheck %s

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sanitizer/lsan_interface.h>

void *p;

int main(int argc, char *argv[]) {
  p = malloc(23);

  assert(__lsan_do_recoverable_leak_check() == 0);

  fprintf(stderr, "Test alloc: %p.\n", malloc(1337));
// CHECK: Test alloc:

  assert(__lsan_do_recoverable_leak_check() == 1);
// CHECK: SUMMARY: {{(Leak|Address)}}Sanitizer: 1337 byte

  // Test that we correctly reset chunk tags.
  p = 0;
  assert(__lsan_do_recoverable_leak_check() == 1);
// CHECK: SUMMARY: {{(Leak|Address)}}Sanitizer: 1360 byte

  _exit(0);
}
