// RUN: %clangxx_asan -O1 -mllvm -asan-use-after-scope=1 %s -o %t && %run %t

#include <stdio.h>
#include <stdlib.h>

int *p[3];

int main() {
  // Variable goes in and out of scope.
  for (int i = 0; i < 3; i++) {
    int x;
    p[i] = &x;
  }
  printf("PASSED\n");
  return 0;
}
