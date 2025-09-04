// Main executable is uninstrumented, but linked to ASan runtime.
// Regression test for https://code.google.com/p/address-sanitizer/issues/detail?id=357.

// RUN: %clangxx -g -O0 %s -c -o %t.o
// RUN: %clangxx_asan -g -O0 %t.o -o %t
// RUN: %run %t 2>&1 | FileCheck %s

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sanitizer/asan_interface.h"
#if __has_feature(ptrauth_calls)
#  include <ptrauth.h>
#endif

void test_shadow(char *p, size_t size) {
  fprintf(stderr, "p = %p\n", p);
  char *q = (char *)__asan_region_is_poisoned(p, size);
  fprintf(stderr, "=%zd=\n", q ? q - p : -1);
}

int main(int argc, char *argv[]) {
  char *p = (char *)malloc(10000);
  test_shadow(p, 100);
  free(p);
  // CHECK: =-1=

  char *mainptr;
#if __has_feature(ptrauth_calls)
  mainptr = (char *)ptrauth_strip((void *)&main, ptrauth_key_return_address);
#else
  mainptr = (char *)&main;
#endif
  test_shadow(mainptr, 1);
  // CHECK: =-1=

  test_shadow((char *)&p, 1);
  // CHECK: =-1=

  return 0;
}
