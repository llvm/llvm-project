// RUN: %clang_dfsan %s -o %t && DFSAN_OPTIONS=fast16labels=1 %run %t
//
// Tests DFSAN_OPTIONS=fast16labels=1
//
#include <sanitizer/dfsan_interface.h>

#include <assert.h>
#include <stdio.h>

int foo(int a, int b) {
  return a + b;
}

int main() {
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
