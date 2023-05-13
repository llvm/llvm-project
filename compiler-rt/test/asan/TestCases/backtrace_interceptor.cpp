// RUN: %clangxx_asan -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s

// Windows does not have execinfo.h. For now, be conservative and
// restrict the test to glibc.
// REQUIRES: glibc-2.27

// Interceptor can cause use-after-free
// (https://github.com/google/sanitizers/issues/321)
// XFAIL: *

// Test the backtrace() interceptor.

#include <assert.h>
#include <execinfo.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_BT 100

int main() {
  void **buffer = (void **)malloc(sizeof(void *) * MAX_BT);
  assert(buffer != NULL);
  free(buffer);

  int numEntries = backtrace(buffer, MAX_BT);
  printf("backtrace returned %d entries\n", numEntries);

  // CHECK: use-after-free
  // CHECK: SUMMARY
  return 0;
}
