// Regression test:
// https://code.google.com/p/address-sanitizer/issues/detail?id=257
// RUN: %clangxx_lsan %s -o %t && %run %t 2>&1 | FileCheck %s

#include <stdio.h>

struct T {
  ~T() { printf("~T\n"); }
};

T *t;

int main(int argc, char **argv) {
  t = new T[argc - 1];
  printf("OK\n");
}

// CHECK: OK

