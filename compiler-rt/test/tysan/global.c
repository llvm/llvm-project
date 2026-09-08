// RUN: %clang_tysan -O0 %s -o %t && %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out
#include <stdlib.h>
#include <string.h>

float P;
long L;

int main() {
  *(int *)&P = 5;
  // CHECK: ERROR: TypeSanitizer: type-aliasing-violation
  // CHECK: WRITE of size 4 at {{.*}} with type int accesses an existing object of type float
  // CHECK: {{#0 0x.* in main .*global.c:}}[[@LINE-3]]

  P = *(((float *)&L) + 1);
  // CHECK: ERROR: TypeSanitizer: type-aliasing-violation
  // CHECK: READ of size 4 at {{.*}} with type float accesses part of an existing object of type long that starts at offset -4
  // CHECK: {{#0 0x.* in main .*global.c:}}[[@LINE-3]]

  return 0;
}

// CHECK-NOT: ERROR: TypeSanitizer: type-aliasing-violation
