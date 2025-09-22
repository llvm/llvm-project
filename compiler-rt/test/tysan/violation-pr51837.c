// RUN: %clang_tysan -O0 %s -o %t && %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

#include <stdint.h>
#include <stdio.h>

// CHECK-NOT: TypeSanitizer

union a {
  int16_t b;
  uint64_t c;
} d;

uint64_t *e = &d.c;
static uint16_t f(int16_t a, int32_t b, uint64_t c);
static int64_t g(int32_t aa, uint8_t h, union a bb) {
  int16_t *i = &d.b;
  f(0, h, 0);
  *i = h;
  return 0;
}
uint16_t f(int16_t a, int32_t b, uint64_t c) {
  for (d.c = 0; 0;)
    ;
  *e = 0;
  return 0;
}

int main() {
  uint32_t j = 8;
  g(1, j, d);
  printf("%d\n", d.b);
  return 0;
}
