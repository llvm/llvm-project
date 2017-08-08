// Test the handle_sigill option.
// RUN: %clang %s -o %t -O1
// RUN:                                not --crash %run %t 2>&1 | FileCheck --check-prefix=CHECK0 %s
// RUN: %env_tool_opts=handle_sigill=0 not --crash %run %t 2>&1 | FileCheck --check-prefix=CHECK0 %s
// RUN: %env_tool_opts=handle_sigill=1 not         %run %t 2>&1 | FileCheck --check-prefix=CHECK1 %s
// FIXME: implement in other sanitizers, not just asan.
// XFAIL: msan
// XFAIL: lsan
// XFAIL: tsan
//
// FIXME: seems to fail on ARM
// REQUIRES: x86_64-target-arch
#include <assert.h>
#include <stdio.h>
#include <sanitizer/asan_interface.h>

void death() {
  fprintf(stderr, "DEATH CALLBACK\n");
}

int main(int argc, char **argv) {
  __sanitizer_set_death_callback(death);
  __builtin_trap();
}
// CHECK1: ERROR: {{.*}}Sanitizer:
// CHECK1: DEATH CALLBACK
// CHECK0-NOT: Sanitizer
