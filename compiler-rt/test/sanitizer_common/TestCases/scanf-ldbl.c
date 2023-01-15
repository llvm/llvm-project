// RUN: %clang %s -o %t && %run %t 2>&1

// Issue #41838
// XFAIL: sparc-target-arch && target={{.*solaris.*}}

#include <assert.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
  long double ld;
  memset(&ld, 255, sizeof ld);
  sscanf("4.0", "%Lf", &ld);
  assert(ld == 4.0);
  return 0;
}
