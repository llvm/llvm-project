// Regression test with an empty (length = 0) custom section.

// RUN: %clangxx_asan -g -O0 %s -c -o %t.o
// RUN: %clangxx_asan -g -O0 %t.o -o %t -sectcreate mysegment mysection /dev/null
// RUN: %run %t 2>&1 | FileCheck %s

#include <stdio.h>

int main() {
  printf("Hello, world!\n");
  // CHECK: Hello, world!
}
