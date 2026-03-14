// RUN: %clang_cl_asan %Od %s %Fe%t
// RUN: %run %t | FileCheck %s

#include <stdio.h>

int main() {
  int subscript = 1;
  char buffer[42];
  buffer[subscript] = 42;
  printf("OK\n");
// CHECK: OK
}
