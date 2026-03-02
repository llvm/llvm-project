// RUN: %clang_tysan -O0 %s -o %t && %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

#include <stdio.h>

typedef struct S1 {
  int i1;
} s1;
typedef struct S2 {
  int i2;
} s2;

void g(int *i) {
  *i = 5;
  printf("%i\n", *i);
}

void h(char *c) {
  *c = 5;
  printf("%i\n", (int)*c);
}

void f(s1 *s1p, s2 *s2p) {
  s1p->i1 = 2;
  s2p->i2 = 3;
  // CHECK: ERROR: TypeSanitizer: type-aliasing-violation
  // CHECK: WRITE of size 4 at {{.*}} with type int (in S2 at offset 0) accesses an existing object of type int (in S1 at offset 0)
  // CHECK: {{#0 0x.* in f .*struct.c:}}[[@LINE-3]]
  printf("%i\n", s1p->i1);
}

int main() {
  s1 s = {.i1 = 1};
  f(&s, (s2 *)&s);
  g(&s.i1);
  h((char *)&s.i1);
}

// CHECK-NOT: ERROR: TypeSanitizer: type-aliasing-violation
