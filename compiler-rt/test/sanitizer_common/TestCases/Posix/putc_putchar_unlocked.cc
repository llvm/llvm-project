// RUN: %clangxx -g %s -o %t && %run %t | FileCheck %s
// CHECK: bc

#include <assert.h>
#include <stdio.h>

int main(void) {
  assert(putc_unlocked('b', stdout) != EOF);
  assert(putchar_unlocked('c') != EOF);

  return 0;
}
