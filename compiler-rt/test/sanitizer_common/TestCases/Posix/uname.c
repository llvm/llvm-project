// RUN: %clang %s -o %t && %run %t

#include <assert.h>
#include <stdio.h>
#include <sys/utsname.h>

int main() {
  struct utsname buf;
  int err = uname(&buf);
  assert(err >= 0);
  printf("%s %s %s %s %s\n", buf.sysname, buf.nodename, buf.release,
         buf.version, buf.machine);
}
