// RUN: %clangxx_asan -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s

// Windows does not have execinfo.h. For now, be conservative and
// restrict the test to glibc.
// REQUIRES: glibc-2.27

// Test the backtrace_symbols() interceptor.

#include <assert.h>
#include <execinfo.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_BT 100

int main() {
  void **buffer = (void **)malloc(sizeof(void *) * MAX_BT);
  assert(buffer != NULL);

  int numEntries = backtrace(buffer, MAX_BT);
  printf("backtrace returned %d entries\n", numEntries);

  free(buffer);

  // Deliberate use-after-free of 'buffer'. We expect ASan to
  // catch this, without triggering internal sanitizer errors.
  char **strings = backtrace_symbols(buffer, numEntries);
  assert(strings != NULL);

  for (int i = 0; i < numEntries; i++) {
    printf("%s\n", strings[i]);
  }

  free(strings);

  // CHECK: use-after-free
  // CHECK: SUMMARY
  return 0;
}
