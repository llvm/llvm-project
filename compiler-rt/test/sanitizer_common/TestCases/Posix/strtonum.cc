// RUN: %clangxx -O0 -g %s -o %t && %run %t 2>&1 | FileCheck %s
//
// UNSUPPORTED: linux, darwin, solaris

#define _OPENBSD_SOURCE

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
  const char *errstr;

  printf("strtonum\n");

  long long l = strtonum("100", 1, 100, &errstr);
  assert(!errstr);
  printf("%lld\n", l);

  l = strtonum("200", 1, 100, &errstr);
  assert(errstr);
  printf("%s\n", errstr);

  l = strtonum("300", 1000, 1001, &errstr);
  assert(errstr);
  printf("%s\n", errstr);

  l = strtonum("abc", 1000, 1001, &errstr);
  assert(errstr);
  printf("%s\n", errstr);

  l = strtonum("1000", 1001, 1000, &errstr);
  assert(errstr);
  printf("%s\n", errstr);

  l = strtonum("1000abc", 1000, 1001, &errstr);
  assert(errstr);
  printf("%s\n", errstr);

  l = strtonum("1000.0", 1000, 1001, &errstr);
  assert(errstr);
  printf("%s\n", errstr);

  // CHECK: strtonum
  // CHECK: 100
  // CHECK: too large
  // CHECK: too small
  // CHECK: invalid
  // CHECK: invalid
  // CHECK: invalid
  // CHECK: invalid

  return 0;
}
