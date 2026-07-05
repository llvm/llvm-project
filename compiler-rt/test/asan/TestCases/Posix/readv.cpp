// RUN: %clangxx_asan -O0 %s -o %t && %run %t
// RUN: %clangxx_asan -O0 %s -DPOSITIVE -o %t && not %run %t 2>&1 | FileCheck %s

// Test the readv() interceptor.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/uio.h>
#include <time.h>

int main() {
  char buf[2011];
  struct iovec iov[2];
#ifdef POSITIVE
  char * volatile buf_ = buf;
  iov[0].iov_base = buf_ - 1;
#else
  iov[0].iov_base = buf + 1;
#endif
  iov[0].iov_len = 5;
  iov[1].iov_base = buf + 10;
  iov[1].iov_len = 2000;
  int fd = open("/etc/hosts", O_RDONLY);
  assert(fd > 0);
  readv(fd, iov, 2);
  // CHECK: WRITE of size 5 at
  close(fd);
  return 0;
}
