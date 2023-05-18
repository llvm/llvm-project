// Test the handle_sigfpe option.
// RUN: %clangxx %s -o %t
// RUN:                               not         %run %t 2>&1 | FileCheck --check-prefix=CHECK1 %s
// RUN: %env_tool_opts=handle_sigfpe=0 not --crash %run %t 2>&1 | FileCheck --check-prefix=CHECK0 %s
// RUN: %env_tool_opts=handle_sigfpe=1 not         %run %t 2>&1 | FileCheck --check-prefix=CHECK1 %s

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
  volatile int one = 1;
  volatile int zero = 0;
  volatile int sink;
  sink = one / zero;
}

// CHECK0-NOT: Sanitizer:DEADLYSIGNAL
// CHECK1: ERROR: {{.*}}Sanitizer: FPE
// CHECK1: {{#[0-9]+.* main .*fpe\.cpp}}:[[@LINE-5]]
// CHECK1: DEATH CALLBACK
// CHECK0-NOT: {{.*}}Sanitizer: FPE
