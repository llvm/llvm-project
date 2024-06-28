// RUN: %clang_tysan -O0 %s -o %t && %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

#include <stdio.h>

struct X {
  int a, b, c;
} x;

static struct X xArray[2];

int main() {
  x.a = 1;
  x.b = 2;
  x.c = 3;

  printf("%d %d %d\n", x.a, x.b, x.c);
  // CHECK-NOT: ERROR: TypeSanitizer: type-aliasing-violation

  for (size_t i = 0; i < 2; i++) {
    xArray[i].a = 1;
    xArray[i].b = 1;
    xArray[i].c = 1;
  }

  struct X *xPtr = (struct X *)&(xArray[0].c);
  xPtr->a = 1;
  // CHECK: ERROR: TypeSanitizer: type-aliasing-violation
  // CHECK: WRITE of size 4 at {{.*}} with type int (in X at offset 0) accesses an existing object of type int (in X at offset 8)
  // CHECK: {{#0 0x.* in main .*struct-members.c:}}[[@LINE-3]]
}
