// RUN: %clang_tysan %s -o %t && %run %t 10 >%t.out.0 2>&1
// RUN: FileCheck --check-prefixes=CHECK,CHECK-BOTH %s < %t.out.0
// RUN: echo "fun:typeViolationignored" > %tmp
// RUN: echo "src:*ignorelist.h" > %tmp
// RUN: %clang_tysan -fsanitize-ignorelist=%tmp %s -o %t && %run %t 10 >%t.out 2>&1
// RUN: FileCheck --check-prefixes=CHECK-IGNORELIST,CHECK-BOTH %s < %t.out

#include "ignorelist.h"
#include <stdio.h>
#include <stdlib.h>

void typeViolationIgnored(float *fPtr) { printf("As int: %d\n", *(int *)fPtr); }

void typeViolation(int *fPtr) { printf("As float: %f\n", *(float *)fPtr); }

int main() {
  float *f = (float *)malloc(sizeof(float));
  *f = 413.0f;
  typeViolationIgnored(f);
  // CHECK: TypeSanitizer: type-aliasing-violation on address 0x{{.*}}
  // CHECK-NEXT: READ of size 4 at 0x{{.*}} with type int accesses an existing object of type float
  // CHECK-IGNORELIST-NOT: TypeSanitizer: type-aliasing-violation on address 0x{{.*}}

  int *i = (int *)malloc(sizeof(int));
  *i = 612;
  typeViolation(i);
  // CHECK-BOTH: TypeSanitizer: type-aliasing-violation on address 0x{{.*}}
  // CHECK-BOTH: READ of size 4 at 0x{{.*}} with type float accesses an existing object of type int

  typeViolationMultiFile((void *)i);
  // CHECK: TypeSanitizer: type-aliasing-violation on address 0x{{.*}}
  // CHECK-IGNORELIST-NOT: TypeSanitizer: type-aliasing-violation on address 0x{{.*}}

  return 0;
}
