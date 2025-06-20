// RUN: %clang_tysan -O0 %s -o %t && %run %t >%t.out.0 2>&1
// RUN: FileCheck %s < %t.out.0
// RUN: %clang_tysan -O2 %s -o %t && %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

#include <stdio.h>

// There's no type-based-aliasing violation here: the memcpy is implemented
// using only char* or unsigned char* (both of which may alias anything).
// CHECK-NOT: ERROR: TypeSanitizer: type-aliasing-violation

void my_memcpy_uchar(void *dest, void *src, int n) {
  unsigned char *p = dest, *q = src, *end = p + n;
  while (p < end)
    *p++ = *q++;
}

void my_memcpy_char(void *dest, void *src, int n) {
  char *p = dest, *q = src, *end = p + n;
  while (p < end)
    *p++ = *q++;
}

void test_uchar() {
  struct S {
    short x;
    short *r;
  } s = {10, &s.x}, s2;
  my_memcpy_uchar(&s2, &s, sizeof(struct S));
  printf("%d\n", *(s2.r));
}

void test_char() {
  struct S {
    short x;
    short *r;
  } s = {10, &s.x}, s2;
  my_memcpy_char(&s2, &s, sizeof(struct S));
  printf("%d\n", *(s2.r));
}

int main() {
  test_uchar();
  test_char();
}
