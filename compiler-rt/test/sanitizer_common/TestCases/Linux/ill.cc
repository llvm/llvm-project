// Test the handle_sigill option.

// RUN: %clangxx %s -o %t -O1
// RUN:                                not --crash %run %t 2>&1 | FileCheck --check-prefix=CHECK0 %s
// RUN: %env_tool_opts=handle_sigill=0 not --crash %run %t 2>&1 | FileCheck --check-prefix=CHECK0 %s
// RUN: %env_tool_opts=handle_sigill=1 not         %run %t 2>&1 | FileCheck --check-prefix=CHECK1 %s

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

// CHECK0-NOT: Sanitizer:DEADLYSIGNAL
// CHECK1: ERROR: {{.*}}Sanitizer: ILL
// CHECK1: {{#[0-9]+.* main .*ill\.cc:[0-9]+}}
// CHECK1: DEATH CALLBACK
// CHECK0-NOT: Sanitizer
