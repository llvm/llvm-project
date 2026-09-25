// RUN: %clang_cl -MD -c -o %t %s
// RUN: %llvm_jitlink %t 2>&1 | FileCheck %s
// CHECK: Entering main
// CHECK-NEXT: Second
// CHECK-NEXT: First

#include <stdio.h>
#include <stdlib.h>

void first(void) {
  printf("First\n");
  fflush(stdout);
}

void second(void) {
  printf("Second\n");
  fflush(stdout);
}

int main(int argc, char *argv[]) {
  atexit(first);
  atexit(second);
  printf("Entering main\n");
  return 0;
}
