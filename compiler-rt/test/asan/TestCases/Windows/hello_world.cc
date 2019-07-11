// RUN: %clang_cl_asan -Od %s -Fe%t
// RUN: %run %t | FileCheck %s

#include <stdio.h>

int main() {
  printf("Hello, world!\n");
// CHECK: Hello, world!
}
