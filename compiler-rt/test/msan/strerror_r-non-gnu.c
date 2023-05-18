// RUN: %clang_msan -std=c99 -O0 -g %s -o %t && %run %t

// strerror_r under a weird set of circumstances can be redirected to
// __xpg_strerror_r. Test that MSan handles this correctly.

#define _POSIX_C_SOURCE 200112
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

int main() {
  char buf[1000];
  int res = strerror_r(EINVAL, buf, sizeof(buf));
  assert(!res);
  volatile int z = strlen(buf);
  return 0;
}
