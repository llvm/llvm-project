// Test for disabling LSan at link-time.
// RUN: %clangxx_lsan %s -o %t
// RUN: %env_lsan_opts=use_stacks=0:use_registers=0 %run %t
// RUN: %env_lsan_opts=use_stacks=0:use_registers=0 not %run %t foo 2>&1 | FileCheck %s

// Fixme: remove once test passes with hwasan
// UNSUPPORTED: hwasan
//
// UNSUPPORTED: darwin

#include <sanitizer/lsan_interface.h>

int argc_copy;

extern "C" {
int __attribute__((used)) __lsan_is_turned_off() {
  return (argc_copy == 1);
}
}

int main(int argc, char *argv[]) {
  volatile int *x = new int;
  *x = 42;
  argc_copy = argc;
  return 0;
}

// CHECK: SUMMARY: {{(Leak|Address)}}Sanitizer: 4 byte(s) leaked in 1 allocation(s)
