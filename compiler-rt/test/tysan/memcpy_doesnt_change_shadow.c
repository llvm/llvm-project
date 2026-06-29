// RUN: %clang_tysan -O0 %s -o %t && %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

#include <stdio.h>
#include <string.h>

int main() {

  int i = 0;
  float f = 5.0f;

  *(float *)&i = 3.0f;
  *(int *)&f = 2;
  // CHECK: WRITE of size 4 at 0x{{.*}} with type float accesses an existing object of type int
  // CHECK: WRITE of size 4 at 0x{{.*}} with type int accesses an existing object of type float

  memcpy(&i, &f, sizeof(int));

  *(float *)&i = 3.0f;
  *(int *)&f = 2;
  // CHECK: WRITE of size 4 at 0x{{.*}} with type float accesses an existing object of type int
  // CHECK: WRITE of size 4 at 0x{{.*}} with type int accesses an existing object of type float

  return 0;
}
