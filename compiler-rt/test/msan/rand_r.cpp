// RUN: %clangxx_msan -O0 -g %s -o %t && %run %t
// RUN: %clangxx_msan -O0 -g -DUNINIT %s -o %t && not %run %t 2>&1 | FileCheck %s

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
  unsigned seed;
#ifndef UNINIT
  seed = 42;
#endif
  int v = rand_r(&seed);
  // CHECK: MemorySanitizer: use-of-uninitialized-value
  // CHECK: in main{{.*}}rand_r.cpp:[[@LINE-2]]
  if (v) printf(".\n");
  return 0;
}
