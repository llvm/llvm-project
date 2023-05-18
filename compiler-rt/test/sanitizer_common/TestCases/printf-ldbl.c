// RUN: %clang %s -o %t && %run %t 2>&1

// Issue #41838
// XFAIL: sparc-target-arch && target={{.*solaris.*}}

#include <assert.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
  char buf[20];
  long double ld = 4.0;
  snprintf(buf, sizeof buf, "%Lf %d", ld, 123);
  assert(!strcmp(buf, "4.000000 123"));
  return 0;
}
