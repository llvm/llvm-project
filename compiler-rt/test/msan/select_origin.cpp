// RUN: %clangxx_msan -fsanitize-memory-track-origins -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_msan -fsanitize-memory-track-origins -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_msan -fsanitize-memory-track-origins -O2 %s -o %t && not %run %t 2>&1 | FileCheck %s

// Test condition origin propagation through "select" IR instruction.

#include <stdio.h>
#include <stdint.h>

__attribute__((noinline))
int *max_by_ptr(int *a, int *b) {
  return *a < *b ? b : a;
}

int main(void) {
  int x;
  int *volatile px = &x;
  int y = 43;
  int *p = max_by_ptr(px, &y);
  // CHECK: Uninitialized value was created by an allocation of 'x' in the stack frame
  return *p;
}
