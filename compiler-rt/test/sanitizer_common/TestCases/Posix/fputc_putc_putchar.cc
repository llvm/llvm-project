// RUN: %clangxx -g %s -o %t && %run %t | FileCheck %s
// CHECK: abc

#include <assert.h>
#include <stdio.h>

int main(void) {
  assert(fputc('a', stdout) != EOF);
  assert(putc('b', stdout) != EOF);
  assert(putchar('c') != EOF);

  return 0;
}
