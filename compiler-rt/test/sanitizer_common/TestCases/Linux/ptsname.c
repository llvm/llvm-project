// RUN: %clang %s -o %t && %run %t

#define _GNU_SOURCE
#define _XOPEN_SOURCE 600

#include <assert.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main() {
  int pt = posix_openpt(O_NOCTTY);
  if (pt == -1)
    return 0;
  char *s = ptsname(pt);
  assert(s);
  assert(strstr(s, "/dev"));

  char buff[1000] = {};
  int r = ptsname_r(pt, buff, sizeof(buff));
  assert(!r);
  assert(strstr(buff, "/dev"));

  close(pt);
  return 0;
}
