// RUN: %clangxx_asan -O0 -fsanitize-address-use-after-scope %s -o %t && %run %t

// Function jumps over variable initialization making lifetime analysis
// ambiguous. Asan should ignore such variable and program must not fail.

#include <stdlib.h>

int *ptr;

void f1(int cond) {
  if (cond)
    goto label;
  int tmp;

 label:
  ptr = &tmp;
  *ptr = 5;
}

void f2(int cond) {
  switch (cond) {
  case 1: {
    ++cond;
    int tmp;
    ptr = &tmp;
    exit(0);
  case 2:
    ptr = &tmp;
    *ptr = 5;
    exit(0);
  }
  }
}

void f3(int cond) {
  {
    int tmp;
    goto l2;
  l1:
    ptr = &tmp;
    *ptr = 5;

    exit(0);
  }
 l2:
  goto l1;
}

void use(int *x) {
  static int c = 10;
  if (--c == 0)
    exit(0);
  (*x)++;
}

void f4() {
  {
    int x;
 l2:
    use(&x);
    goto l1;
  }
 l1:
  goto l2;
}

int main() {
  f1(1);
  f2(1);
  f3(1);
  f4();
  return 0;
}
