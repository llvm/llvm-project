// RUN: %clang_tysan -O0 %s -o %t && %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

#include <stdio.h>

// The two anonymous structs are structurally identical. As a result, we don't
// report an aliasing violation here.
// CHECK-NOT: ERROR: TypeSanitizer: type-aliasing-violation

typedef struct {
  int i1;
} s1;
typedef struct {
  int i2;
} s2;

void f(s1 *s1p, s2 *s2p) {
  s1p->i1 = 2;
  s2p->i2 = 3;
  printf("%i\n", s1p->i1);
}

int main() {
  s1 s = {.i1 = 1};
  f(&s, (s2 *)&s);
}
