// RUN: %clangxx -g %s -o %t
// RUN: %env_tool_opts=decorate_proc_maps=1 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-%tool_name

// REQUIRES: stable-runtime
// XFAIL: android && asan

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

bool CopyFdToFd(int in_fd, int out_fd) {
  const size_t kBufSize = 0x10000;
  static char buf[kBufSize];
  while (true) {
    ssize_t got = read(in_fd, buf, kBufSize);
    if (got > 0) {
      write(out_fd, buf, got);
    } else if (got == 0) {
      break;
    } else if (errno != EAGAIN || errno != EWOULDBLOCK || errno != EINTR) {
      fprintf(stderr, "error reading file, errno %d\n", errno);
      return false;
    }
  }
  return true;
}

void *ThreadFn(void *arg) {
  (void)arg;
  int fd = open("/proc/self/maps", O_RDONLY);
  bool res = CopyFdToFd(fd, 2);
  close(fd);
  return (void *)!res;
}

int main(void) {
  pthread_t t;
  void *res;
  pthread_create(&t, 0, ThreadFn, 0);
  pthread_join(t, &res);
  return (int)(size_t)res;
}

// CHECK-asan: rw-p {{.*}} [low shadow]
// CHECK-asan: ---p {{.*}} [shadow gap]
// CHECK-asan: rw-p {{.*}} [high shadow]

// CHECK-hwasan: rw-p {{.*}} [low shadow]
// CHECK-hwasan: ---p {{.*}} [shadow gap]
// CHECK-hwasan: rw-p {{.*}} [high shadow]

// CHECK-msan: ---p {{.*}} [invalid]
// CHECK-msan: rw-p {{.*}} [shadow{{.*}}]
// CHECK-msan: ---p {{.*}} [origin{{.*}}]

// CHECK-tsan: rw-p {{.*}} [shadow]
// CHECK-tsan: rw-p {{.*}} [meta shadow]

// Nothing interesting with standalone LSan and UBSan.
// CHECK-lsan: decorate_proc_maps
// CHECK-ubsan: decorate_proc_maps
