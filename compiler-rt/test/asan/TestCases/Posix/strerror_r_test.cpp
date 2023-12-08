// RUN: %clangxx_asan -O0 %s -o %t && %run %t

// Regression test for PR17138.

#include <assert.h>
#include <string.h>
#include <stdio.h>

int main() {
  char buf[1024];
  char *res = (char *)strerror_r(300, buf, sizeof(buf));
  printf("%p\n", res);
  return 0;
}
