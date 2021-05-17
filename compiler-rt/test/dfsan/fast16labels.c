// RUN: %clang_dfsan %s -mllvm -dfsan-fast-16-labels -o %t
// RUN: %run %t
//
// REQUIRES: x86_64-target-arch
//
// Tests fast16labels mode.

#include <sanitizer/dfsan_interface.h>

#include <assert.h>
#include <stdio.h>
#include <string.h>

int foo(int a, int b) {
  return a + b;
}

int main(int argc, char *argv[]) {
  int a = 10;
  int b = 20;
  dfsan_set_label(8, &a, sizeof(a));
  dfsan_set_label(512, &b, sizeof(b));
  int c = foo(a, b);
  printf("A: 0x%x\n", dfsan_get_label(a));
  printf("B: 0x%x\n", dfsan_get_label(b));
  dfsan_label l = dfsan_get_label(c);
  printf("C: 0x%x\n", l);
  assert(l == 520);  // OR of the other two labels.
}
