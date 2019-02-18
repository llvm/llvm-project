// Test shadow faults during esan initialization as well as
// faults during dlsym's calloc during interceptor init.
//
// RUN: %clang_esan_wset %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s
// Stucks at init and no clone feature equivalent.
// UNSUPPORTED: freebsd

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Our goal is to emulate an instrumented allocator, whose calloc
// invoked from dlsym will trigger shadow faults, to test an
// early shadow fault during esan interceptor init.
// We do this by replacing calloc:
void *calloc(size_t size, size_t n) {
  // Unfortunately we can't print anything to make the test
  // ensure we got here b/c the sanitizer interceptors can't
  // handle that during interceptor init.

  // Ensure we trigger a shadow write fault:
  int x[16];
  x[0] = size;
  // Now just emulate calloc.
  void *res = malloc(size*n);
  memset(res, 0, size*n);
  return res;
}

int main(int argc, char **argv) {
  printf("all done\n");
  return 0;
}
// CHECK: all done
