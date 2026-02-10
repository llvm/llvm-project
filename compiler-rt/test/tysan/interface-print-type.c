// RUN: %clang_tysan %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s

#include <sanitizer/tysan_interface.h>
#include <stdio.h>

struct S {
  int i;
  float f;
};

void printInt(int *i) {
  const int bufferSize = 512;
  static char nameBuffer[bufferSize];
  __tysan_get_type_name(i, nameBuffer, 512);
  printf("%d, %s\n", *i, nameBuffer);
  fflush(stdout);
}

int main() {
  struct S s;
  s.i = 4;
  printInt((int *)&s);
  // CHECK: 4, int (in S at offset 0)

  s.f = 5.0f;
  // CHECK: ERROR: TypeSanitizer: type-aliasing-violation
  // CHECK: READ of size 4 at 0x{{.*}} with type int accesses an existing object of type float (in S at offset 4)
  // CHECK: {{.*}}, float (in S at offset 4)
  printInt((int *)&s.f);

  return 0;
}
