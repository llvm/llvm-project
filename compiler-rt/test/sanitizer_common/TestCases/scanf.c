// RUN: %clang -std=c17 %s -o %t && %run %t
/// Test __isoc23_* for glibc 2.38+.
// RUN: %clang -std=c2x %s -o %t && %run %t

#include <assert.h>
#include <stdarg.h>
#include <stdio.h>

int test_vsscanf(const char *buf, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  int ret = vsscanf(buf, fmt, ap);
  va_end(ap);
  return ret;
}

int main(int argc, char **argv) {
  int x, y;
  assert(sscanf("42", "%d", &x) == 1);
  assert(x == 42);
  assert(test_vsscanf("42", "%d", &y) == 1);
  assert(y == 42);
  return 0;
}
