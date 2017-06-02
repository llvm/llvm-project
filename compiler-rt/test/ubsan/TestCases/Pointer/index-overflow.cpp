// RUN: %clangxx -fsanitize=pointer-overflow %s -o %t
// RUN: %t 1 2>&1 | FileCheck %s --check-prefix=ERR
// RUN: %t 0 2>&1 | FileCheck %s --check-prefix=SAFE
// RUN: %t -1 2>&1 | FileCheck %s --check-prefix=SAFE

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  // SAFE-NOT: runtime error
  // ERR: runtime error: pointer index expression with base {{.*}} overflowed to

  char *p = (char *)(UINTPTR_MAX);

  printf("%p\n", p + atoi(argv[1]));

  return 0;
}
