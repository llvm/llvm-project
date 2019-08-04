// RUN: %clang_msan -O0 -g %s -o %t && %run %t

#include <assert.h>
#include <errno.h>
#include <string.h>

int main() {
  char buf[1000];
  char *res = strerror_r(EINVAL, buf, sizeof(buf));
  assert(res);
  volatile int z = strlen(res);

  res = strerror_r(-1, buf, sizeof(buf));
  assert(res);
  z = strlen(res);

  return 0;
}
