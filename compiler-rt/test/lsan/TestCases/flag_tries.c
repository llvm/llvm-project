// Test retries option of lsan.
// RUN: %clang_lsan %s -o %t
// RUN: %env_lsan_opts=use_stacks=0:use_registers=0:symbolize=0 %run %t foo 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK1
// RUN: %env_lsan_opts=use_stacks=0:use_registers=0:symbolize=0:tries=12 %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK12

#include <assert.h>
#include <sanitizer/lsan_interface.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

void *p;

int main(int argc, char *argv[]) {
  fprintf(stderr, "Test alloc: %p.\n", malloc(1337));
  // CHECK: Test alloc:

  assert(__lsan_do_recoverable_leak_check() == 1);
  // CHECK1-COUNT-1: SUMMARY: {{.*}}Sanitizer: 1337 byte
  // CHECK12-COUNT-12: SUMMARY: {{.*}}Sanitizer: 1337 byte

  _exit(0);
}
