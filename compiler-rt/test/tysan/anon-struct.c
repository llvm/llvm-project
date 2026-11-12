// RUN: %clang_tysan -O0 %s -o %t && %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

#include <stdio.h>

typedef struct {
  int i1, i1b;
} s1;
typedef struct {
  int i2, i2b, i2c;
} s2;

void f(s1 *s1p, s2 *s2p) {
  s1p->i1 = 2;
  s2p->i2 = 3;
  // CHECK: ERROR: TypeSanitizer: type-aliasing-violation
  // CHECK: WRITE of size 4 at {{.*}} with type int (in <anonymous type> at offset 0) accesses an existing object of type int (in <anonymous type> at offset 0)
  // CHECK: {{#0 0x.* in f .*anon-struct.c:}}[[@LINE-3]]
  printf("%i\n", s1p->i1);
}

int main() {
  s1 s = {.i1 = 1, .i1b = 5};
  f(&s, (s2 *)&s);
}

// CHECK-NOT: ERROR: TypeSanitizer: type-aliasing-violation
