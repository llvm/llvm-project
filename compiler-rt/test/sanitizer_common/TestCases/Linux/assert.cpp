// Test the handle_abort option.

// RUN: %clangxx %s -o %t
// RUN:                              not --crash %run %t 2>&1 | FileCheck --check-prefix=CHECK0 %s
// RUN: %env_tool_opts=handle_abort=0 not --crash %run %t 2>&1 | FileCheck --check-prefix=CHECK0 %s
// RUN: %env_tool_opts=handle_abort=1 not         %run %t 2>&1 | FileCheck --check-prefix=CHECK1 %s

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

// CHECK0-NOT: Sanitizer:DEADLYSIGNAL
// CHECK1: ERROR: {{.*}}Sanitizer: ABRT
// CHECK1: {{ #0 }}
// CHECK1: DEATH CALLBACK
// CHECK0-NOT: Sanitizer
