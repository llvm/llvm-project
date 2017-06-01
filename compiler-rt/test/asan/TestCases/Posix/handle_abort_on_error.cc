// Regression test: this used to abort() in SIGABRT handler in an infinite loop.
// RUN: %clangxx_asan -O0 %s -o %t && %env_asan_opts=handle_abort=1,abort_on_error=1 not --crash %run %t 2>&1 | FileCheck %s

#include <stdlib.h>

int main() {
  abort();
  // CHECK: ERROR: AddressSanitizer: ABRT
}
