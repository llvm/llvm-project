// Test that the preloaded runtime works without linking the static library.

// RUN: %clang %s -lstdc++ -o %t
// RUN: env LD_PRELOAD=%shared_libscudo not %run %t 2>&1 | FileCheck %s

// This way of setting LD_PRELOAD does not work with Android test runner.
// REQUIRES: !android

#include <assert.h>

int main(int argc, char *argv[]) {
  int *p = new int;
  assert(p);
  *p = 0;
  delete p;
  delete p;
  return 0;
}

// CHECK: ERROR: invalid chunk state
