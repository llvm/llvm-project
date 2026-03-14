// RUN: %clangxx_msan -O0 -g %s -o %t && %run %t

#define _BSD_SOURCE
#include <assert.h>
#include <stdlib.h>

#include <sanitizer/msan_interface.h>

int main(void) {
  double x[4];
  int ret = getloadavg(x, 3);
  assert(ret > 0);
  assert(ret <= 3);
  assert(__msan_test_shadow(x, sizeof(double) * ret) == -1);
  assert(__msan_test_shadow(&x[ret], sizeof(double)) == 0);
}
