// Test that realloc(nullptr, 0) return a non-NULL pointer.

// RUN: %clang_tsan %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s

#include <malloc/malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>

int main() {
  void *p = realloc(nullptr, 0);
  if (!p) {
    abort();
  }
  fprintf(stderr, "Okay.\n");
  return 0;
}

// CHECK: Okay.
