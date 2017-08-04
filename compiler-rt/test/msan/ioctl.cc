// RUN: %clangxx_msan -O0 -g %s -o %t && %run %t
// RUN: %clangxx_msan -O3 -g %s -o %t && %run %t

#include <assert.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

int main(int argc, char **argv) {
  int fd = socket(AF_UNIX, SOCK_DGRAM, 0);

  unsigned int z;
  int res = ioctl(fd, FIOGETOWN, &z);
  assert(res == 0);
  close(fd);
  if (z)
    exit(0);
  return 0;
}
