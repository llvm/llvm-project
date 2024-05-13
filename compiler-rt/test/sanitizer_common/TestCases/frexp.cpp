// RUN: %clangxx -O2 %s -o %t && %run %t 2>&1 | FileCheck %s

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
  for (int i = 0; i < 10000; i++) {
    volatile double x = 10;
    int exp = 0;
    double y = frexp(x, &exp);
    if (y != 0.625 || exp != 4) {
      printf("i=%d y=%lf exp=%d\n", i, y, exp);
      exit(1);
    }
  }
  fprintf(stderr, "DONE\n");
  // CHECK: DONE
  return 0;
}
