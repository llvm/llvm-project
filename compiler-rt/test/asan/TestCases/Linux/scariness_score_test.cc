// Test how we produce the scariness score.
// Linux-specific variant which tests abort() calls. On OS X the process
// disappears before being able to print the scariness.

// RUN: %clangxx_asan -O0 %s -o %t
// RUN: export %env_asan_opts=detect_stack_use_after_return=1:handle_abort=1:print_scariness=1
// RUN: not %run %t 2>&1 | FileCheck %s
// REQUIRES: shell
#include <stdlib.h>

int main(int argc, char **argv) {
  abort();
  // CHECK: SCARINESS: 10 (signal)
}
