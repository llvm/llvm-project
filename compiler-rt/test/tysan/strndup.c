// RUN: %clang_tysan -O0 %s -o %t && %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out
#include <stdlib.h>
#include <string.h>

int main() {
  long val = 32167;
  char *p = strndup((const char *)&val, sizeof(val));
  // CHECK: ERROR: TypeSanitizer: type-aliasing-violation
  // CHECK: READ of size 4 at {{.*}} with type int accesses an existing object of type long
  int inc = ((*(int *)p) + 1);
  free(p);
  return 0;
}
