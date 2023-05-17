// RUN: %clangxx %s -o %t -fexperimental-sanitize-metadata=all && %t | FileCheck %s

// Test that the compiler emits weak declarations to the callbacks, which are
// not called if they do not exist.

#include <stdio.h>

int main() {
  printf("main\n");
  return 0;
}

// CHECK: main
