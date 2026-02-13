// RUN: %clang_tysan -mllvm -tysan-outline-instrumentation=true -O0 %s -o %t && %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out
// RUN: %clang_tysan -mllvm -tysan-outline-instrumentation=true -mllvm -tysan-verify-outlined-instrumentation=true -O0 %s -o %t && %run %t >%t.out 2>&1
// RUN: FileCheck %s --check-prefixes='CHECK,CHECK-VERIFY' < %t.out

#include <stdio.h>
#include <stdlib.h>

struct X {
  int i;
  int j;
};

int foo(struct X *p, struct X *q) {
  q->j = 1;
  p->i = 0;
  // CHECK: ERROR: TypeSanitizer: type-aliasing-violation
  // CHECK-NEXT: WRITE of size 4 at {{.*}} with type int (in X at offset 0) accesses an existing object of type int (in X at offset 4)
  // CHECK-NEXT: {{#0 0x.* in foo .*struct-offset-outline.c:}}[[@LINE-3]]
  // CHECK-VERIFY-EMPTY:
  // CHECK-VERIFY-NEXT: ERROR: TypeSanitizer: type-aliasing-violation
  // CHECK-VERIFY-NEXT: WRITE of size 4 at {{.*}} with type int (in X at offset 0) accesses an existing object of type int (in X at offset 4)
  // CHECK-VERIFY-NEXT: {{#0 0x.* in foo .*struct-offset-outline.c:}}[[@LINE-7]]
  return q->j;
}

int main() {
  unsigned char *p = malloc(3 * sizeof(int));
  printf("%i\n", foo((struct X *)(p + sizeof(int)), (struct X *)p));
}

// CHECK-NOT: ERROR: TypeSanitizer: type-aliasing-violation
