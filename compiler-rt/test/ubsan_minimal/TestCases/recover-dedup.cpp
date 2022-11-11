// RUN: %clangxx -w -fsanitize=signed-integer-overflow,nullability-return,returns-nonnull-attribute -fsanitize-recover=all %s -o %t && %run %t 2>&1 | FileCheck %s

#include <stdint.h>
#include <stdio.h>

int *_Nonnull h() {
  // CHECK: nullability-return by 0x{{[[:xdigit:]]+$}}
  return NULL;
}

__attribute__((returns_nonnull))
int *i() {
  // CHECK: nonnull-return by 0x{{[[:xdigit:]]+$}}
  return NULL;
}

__attribute__((noinline))
int f(int x, int y) {
  // CHECK: mul-overflow by 0x{{[[:xdigit:]]+$}}
  return x * y;
}

__attribute__((noinline))
int g(int x, int y) {
  // CHECK: mul-overflow by 0x{{[[:xdigit:]]+$}}
  return x * (y + 1);
}

int main() {
  h();
  i();
  int x = 2;
  for (int i = 0; i < 10; ++i)
    x = f(x, x);
  x = 2;
  for (int i = 0; i < 10; ++i)
    x = g(x, x);
  // CHECK-NOT: mul-overflow
}
