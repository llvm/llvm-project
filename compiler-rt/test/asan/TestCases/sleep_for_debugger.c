// RUN: %clang_asan -O2 %s -o %t
// RUN: %env_asan_opts=verbosity=1:sleep_before_init=1:sleep_after_init=1:sleep_before_dying=1 not %run %t 2>&1 | FileCheck %s

#include <stdlib.h>

int main() {
  char *x = (char*)malloc(10 * sizeof(char));
  free(x);
  return x[5];
}

// CHECK: Sleeping for 1 second(s) before init
// CHECK: AddressSanitizer Init done
// CHECK: Sleeping for 1 second(s) after init
// CHECK: ERROR: AddressSanitizer
// CHECK: ABORTING
// CHECK: Sleeping for 1 second(s) before dying
