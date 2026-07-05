// Checks if a user pointer is found by the leak sanitizer.
// RUN: %clang_lsan %s -o %t
// RUN: %run %t 2>&1

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

uintptr_t glob[1024];

int main() {
  for (int i = 0; i < 1024; ++i) {
    // Check that the pointers will not be falsely reported as leaks.
    glob[i] = (uintptr_t)malloc(sizeof(int *));
  }
  return 0;
}
