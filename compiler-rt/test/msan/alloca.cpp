// RUN: %clangxx_msan -O0 -g %s -o %t && %run %t
// RUN: %clangxx_msan -O3 -g %s -o %t && %run %t

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sanitizer/msan_interface.h>

int main(void) {
  char *p = (char *)alloca(16);
  assert(0 == __msan_test_shadow(p, 16));
  assert(0 == __msan_test_shadow(p + 15, 1));

  memset(p, 0, 16);
  assert(-1 == __msan_test_shadow(p, 16));

  volatile int x = 0;
  char * volatile q = (char *)alloca(42 * x);
  assert(-1 == __msan_test_shadow(p, 16));

  int r[x];
  int *volatile r2 = r;
  assert(-1 == __msan_test_shadow(p, 16));
}
