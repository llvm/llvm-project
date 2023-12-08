// Test for on-demand leak checking.
// RUN: %clangxx_lsan %s -o %t
// RUN: %env_lsan_opts=use_stacks=0:use_registers=0 %run %t foo 2>&1 | FileCheck %s
// RUN: %env_lsan_opts=use_stacks=0:use_registers=0 %run %t 2>&1 | FileCheck %s

// UNSUPPORTED: darwin

// FIXME: Investigate.
// XFAIL: internal_symbolizer && lsan-standalone && i386-linux

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
// CHECK: SUMMARY: {{.*}}Sanitizer: 1337 byte

  // Test that we correctly reset chunk tags.
  p = 0;
  assert(__lsan_do_recoverable_leak_check() == 1);
// CHECK: SUMMARY: {{.*}}Sanitizer: 1360 byte

  _exit(0);
}
