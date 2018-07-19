// Test that a stack overflow fails as expected

// RUN: %clang_noscs %s -o %t -DITERATIONS=3
// RUN: %run %t | FileCheck %s
// RUN: %clang_noscs %s -o %t -DITERATIONS=12
// RUN: %run %t | FileCheck -check-prefix=OVERFLOW_SUCCESS %s

// RUN: %clang_scs %s -o %t -DITERATIONS=3
// RUN: %run %t | FileCheck %s

// The behavioral check for SCS + overflow lives in the tests overflow-x86_64.c
// and overflow-aarch64.c. This is because the expected behavior is different
// between the two platforms. On x86_64 we crash because the comparison between
// the shadow call stack and the regular stack fails. On aarch64 there is no
// comparison, we just load the return address from the shadow call stack. So we
// just expect not to see the output from print_and_exit.

#include <stdio.h>
#include <stdlib.h>

#include "minimal_runtime.h"

void print_and_exit(void) {
// CHECK-NOT: Stack overflow successful.
// OVERFLOW_SUCCESS: Stack overflow successful.
  scs_fputs_stdout("Stack overflow successful.\n");
  exit(0);
}

int scs_main(void)
{
  void *addrs[4];
  for (int i = 0; i < ITERATIONS; i++)
    addrs[i] = &print_and_exit;

  scs_fputs_stdout("Returning.\n");

  return 0;
}
