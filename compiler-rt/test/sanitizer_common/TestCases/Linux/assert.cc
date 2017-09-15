// Test the handle_abort option.

// clang-format off
// RUN: %clang %s -o %t
// RUN:                              not --crash %run %t 2>&1 | FileCheck --check-prefix=CHECK0 %s
// RUN: %env_tool_opts=handle_abort=0 not --crash %run %t 2>&1 | FileCheck --check-prefix=CHECK0 %s
// RUN: %env_tool_opts=handle_abort=1 not         %run %t 2>&1 | FileCheck --check-prefix=CHECK1 %s
// clang-format on

// FIXME: implement in other sanitizers, not just asan.
// XFAIL: msan
// XFAIL: lsan
// XFAIL: tsan
// XFAIL: ubsan
#include <assert.h>
#include <stdio.h>
#include <sanitizer/asan_interface.h>

void death() {
  fprintf(stderr, "DEATH CALLBACK\n");
}

int main(int argc, char **argv) {
  __sanitizer_set_death_callback(death);
  assert(argc == 100);
}
// CHECK1: ERROR: {{.*}}Sanitizer:
// CHECK1: DEATH CALLBACK
// CHECK0-NOT: Sanitizer
