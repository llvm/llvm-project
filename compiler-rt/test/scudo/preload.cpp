// Test that the preloaded runtime works without linking the static library.

// RUN: %clang %s -o %t
// RUN: env LD_PRELOAD=%shared_libscudo not %run %t 2>&1 | FileCheck %s

// This way of setting LD_PRELOAD does not work with Android test runner.
// REQUIRES: !android

#include <assert.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  void *p = malloc(sizeof(int));
  assert(p);
  free(p);
  free(p);
  return 0;
}

// CHECK: ERROR: invalid chunk state
