// RUN: %clangxx -O0 %s -o %t

// REQUIRES: glibc

#include <assert.h>
#include <fcntl.h>
#include <sys/uio.h>
#include <unistd.h>

int main(void) {
  int fd = open("/proc/self/stat", O_RDONLY);
  char bufa[7];
  char bufb[7];
  struct iovec vec[2];
  vec[0].iov_base = bufa + 4;
  vec[0].iov_len = 1;
  vec[1].iov_base = bufb;
  vec[1].iov_len = sizeof(bufb);
  ssize_t rd = preadv2(fd, vec, 2, 0, 0);
  assert(rd > 0);
  vec[0].iov_base = bufa;
  rd = preadv2(fd, vec, 2, 0, 0);
  assert(rd > 0);
  rd = preadv2(fd, vec, 5, -25, 0);
  assert(rd < 0);
  close(fd);
  return 0;
}
