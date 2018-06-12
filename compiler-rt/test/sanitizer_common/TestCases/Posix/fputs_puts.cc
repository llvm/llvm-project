// RUN: %clangxx -g %s -o %t && %run %t | FileCheck %s
// CHECK: {{^foobar$}}

#include <stdio.h>

int main(void) {
  int r;

  r = fputs("foo", stdout);
  if (r < 0)
    return 1;

  r = puts("bar");
  if (r < 0)
    return 1;

  return 0;
}
