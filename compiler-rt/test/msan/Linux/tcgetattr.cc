// RUN: %clangxx_msan -O0 %s -o %t && %run %t %p

// XFAIL: target-is-mips64el                                                      

#include <assert.h>
#include <glob.h>
#include <stdio.h>
#include <stdlib.h>
#include <termios.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
  int fd = getpt();
  assert(fd >= 0);
  
  struct termios t;
  int res = tcgetattr(fd, &t);
  assert(!res);

  if (t.c_iflag == 0)
    exit(0);
  return 0;
}
