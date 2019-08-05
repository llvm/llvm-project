// RUN: %clangxx -g %s -o %t && %run %t | FileCheck %s
// CHECK: {{^foobar$}}

#include <assert.h>
#include <stdio.h>

int main(void) {
  assert(fputs("foo", stdout) >= 0);
  assert(puts("bar") >= 0);

  return 0;
}
