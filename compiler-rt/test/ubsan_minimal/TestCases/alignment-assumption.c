// RUN: %clang -fsanitize=alignment %s -o %t && %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK

#include <stdlib.h>

int main(int argc, char *argv[]) {

  char *ptr = (char *)malloc(2);

  void *t = __builtin_assume_aligned(ptr + 1, 0x8000);
  (void)t;
  // CHECK: ubsan: alignment-assumption
  // CHECK-NOT: alignment-assumption

  free(ptr);

  return 0;
}
