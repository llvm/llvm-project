// RUN: %clangxx_asan -O0 -g %s -o %t && %env_asan_opts=handle_ioctl=1 not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O3 -g %s -o %t && %env_asan_opts=handle_ioctl=1 not %run %t 2>&1 | FileCheck %s

// RUN: %clangxx_asan -O0 -g %s -o %t && %run %t
// RUN: %clangxx_asan -O3 -g %s -o %t && %run %t

#include <assert.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

#if defined(__sun__) && defined(__svr4__)
#include <sys/filio.h>
#endif

int main(int argc, char **argv) {
  int fd = socket(AF_INET, SOCK_DGRAM, 0);

  int nonblock;
  int res = ioctl(fd, FIONBIO, &nonblock + 1);
  // CHECK: AddressSanitizer: stack-buffer-overflow
  // CHECK: READ of size 4 at
  // CHECK: {{#.* in main .*ioctl.cpp:}}[[@LINE-3]]
  assert(res == 0);
  close(fd);
  return 0;
}
