// RUN: %clang_cl -MD -c -o %t %s
// RUN: %llvm_jitlink %t 2>&1 | FileCheck %s
// CHECK: Hello, world!

#include <stdio.h>
int main(int argc, char *argv[]) {
  printf("Hello, world!\n");
  fflush(stdout);
  return 0;
}
