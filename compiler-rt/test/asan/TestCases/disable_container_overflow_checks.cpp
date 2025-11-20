// Test crash gives guidance on -D__SANITIZER_DISABLE_CONTAINER_OVERFLOW__ and
// ASAN_OPTIONS=detect_container_overflow=0
// RUN: %clangxx_asan -O %s -o %t
// RUN: not %run %t 2>&1 | FileCheck --check-prefix=CHECK-CRASH %s
//
// Test overflow checks can be disabled at runtime with
// ASAN_OPTIONS=detect_container_overflow=0
// RUN: %env_asan_opts=detect_container_overflow=0 %run %t 2>&1 | FileCheck --check-prefix=CHECK-NOCRASH %s
//
// Illustrate use of -D__SANITIZER_DISABLE_CONTAINER_OVERFLOW__ flag to suppress
// overflow checks at compile time.
// RUN: %clangxx_asan -D__SANITIZER_DISABLE_CONTAINER_OVERFLOW__ -O %s -o %t-no-overflow
// RUN: %run %t-no-overflow 2>&1 | FileCheck --check-prefix=CHECK-NOCRASH %s
//
// UNSUPPORTED: ios, android

#include <assert.h>
#include <stdio.h>
#include <string.h>

// public definition of __sanitizer_annotate_contiguous_container
#include "sanitizer/common_interface_defs.h"

static volatile int one = 1;

int TestCrash() {
  long t[100];
  t[60] = 0;
#if __has_feature(address_sanitizer)
  __sanitizer_annotate_contiguous_container(&t[0], &t[0] + 100, &t[0] + 100,
                                            &t[0] + 50);
#endif
  // CHECK-CRASH: AddressSanitizer: container-overflow
  // CHECK-CRASH: ASAN_OPTIONS=detect_container_overflow=0
  // CHECK-CRASH: __SANITIZER_DISABLE_CONTAINER_OVERFLOW__
  // CHECK-NOCRASH-NOT: AddressSanitizer: container-overflow
  // CHECK-NOCRASH-NOT: ASAN_OPTIONS=detect_container_overflow=0
  // CHECK-NOCRASH-NOT: __SANITIZER_DISABLE_CONTAINER_OVERFLOW__
  return (int)t[60 * one]; // Touches the poisoned memory.
}

int main(int argc, char **argv) {

  int retval = 0;

  retval = TestCrash();

  printf("Exiting main\n");

  return retval;
}
