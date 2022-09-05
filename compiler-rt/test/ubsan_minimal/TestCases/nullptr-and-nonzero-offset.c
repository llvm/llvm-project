// RUN: %clang   -fsanitize=pointer-overflow %s -o %t && %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-C --implicit-check-not="pointer-overflow"
// RUN: %clangxx -x c++ -fsanitize=pointer-overflow %s -o %t && %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-CPP --implicit-check-not="pointer-overflow"

#include <stdlib.h>

int main(int argc, char *argv[]) {
  char *base, *result;

  base = (char *)0;
  result = base + 0;
  // CHECK-C: pointer-overflow by 0x{{[[:xdigit:]]+$}}
  // CHECK-CPP-NOT: pointer-overflow

  base = (char *)0;
  result = base + 1;
  // CHECK: pointer-overflow by 0x{{[[:xdigit:]]+$}}

  base = (char *)1;
  result = base - 1;
  // CHECK: pointer-overflow by 0x{{[[:xdigit:]]+$}}

  return 0;
}
