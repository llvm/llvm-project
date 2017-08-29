// RUN: %clangxx -fsanitize=signed-integer-overflow -fsanitize-recover=all %s -o %t && %run %t 2>&1 | FileCheck %s

#include <stdint.h>

__attribute__((noinline))
int f(int x, int y) {
  // CHECK: mul-overflow
  return x * y;
}

__attribute__((noinline))
int g(int x, int y) {
  // CHECK: mul-overflow
  return x * (y + 1);
}

int main() {
  int x = 2;
  for (int i = 0; i < 10; ++i)
    x = f(x, x);
  x = 2;
  for (int i = 0; i < 10; ++i)
    x = g(x, x);
  // CHECK-NOT: mul-overflow
}
