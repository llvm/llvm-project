// RUN: %clangxx -fsanitize=pointer-overflow %s -o %t
// RUN: %run %t  2 2>&1 | FileCheck %s --implicit-check-not="runtime error:" --check-prefix=ERR2
// RUN: %run %t  1 2>&1 | FileCheck %s --implicit-check-not="runtime error:" --check-prefix=ERR1
// RUN: %run %t  0 2>&1 | FileCheck %s --implicit-check-not="runtime error:" --check-prefix=SAFE
// RUN: %run %t -1 2>&1 | FileCheck %s --implicit-check-not="runtime error:" --check-prefix=SAFE
// RUN: %run %t -2 2>&1 | FileCheck %s --implicit-check-not="runtime error:" --check-prefix=SAFE

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  // SAFE-NOT: runtime error
  // ERR2: runtime error: pointer index expression with base {{.*}} overflowed to
  // ERR2: runtime error: pointer index expression with base {{.*}} overflowed to
  // ERR1: runtime error: applying non-zero offset to non-null pointer 0x{{.*}} produced null pointer
  // ERR1: runtime error: applying non-zero offset to non-null pointer 0x{{.*}} produced null pointer

  char *p = (char *)(UINTPTR_MAX);

  printf("%p\n", p + atoi(argv[1]));

  char *q = (char *)(UINTPTR_MAX);

  printf("%p\n", p - (-atoi(argv[1])));

  return 0;
}
