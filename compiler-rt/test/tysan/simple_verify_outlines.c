// RUN: %clang_tysan -mllvm -tysan-outline-instrumentation=true -mllvm -tysan-verify-outlined-instrumentation=true %s -o %t && %run %t >%t.out.0 2>&1
// RUN: FileCheck %s < %t.out.0

#include <stdio.h>

void printInt(int *i) { printf("%d\n", *i); }

int main() {

  float value = 5.0f;
  printInt((int *)&value);

  return 0;
}

// CHECK: ERROR: TypeSanitizer: type-aliasing-violation
// CHECK-NEXT: READ of size 4 at {{.*}} with type int accesses an existing object of type float
// CHECK-NEXT: {{#0 0x.* in printInt}}
// CHECK-EMPTY:
// CHECK-NEXT: ERROR: TypeSanitizer: type-aliasing-violation
// CHECK-NEXT: READ of size 4 at {{.*}} with type int accesses an existing object of type float
// CHECK-NEXT: {{#0 0x.* in printInt}}
