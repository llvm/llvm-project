// RUN: %clang_hwasan -g %s -o %t
// RUN: %env_hwasan_opts=decorate_proc_maps=1 %run %t 2>&1 | FileCheck %s --check-prefix=A
// RUN: %env_hwasan_opts=decorate_proc_maps=1 %run %t 2>&1 | FileCheck %s --check-prefix=B

// A:      rw-p {{.*}}hwasan threads]
// A-NEXT: ---p {{.*}}shadow gap]
// A-NEXT: rw-p {{.*}}low shadow]
// A-NEXT: ---p {{.*}}shadow gap]
// A-NEXT: rw-p {{.*}}high shadow]

// B-DAG: rw-p {{.*}}SizeClassAllocator: region info]
// B-DAG: rw-p {{.*}}LargeMmapAllocator]
// B-DAG: rw-p {{.*}}stack depot]

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <pthread.h>
#include <stdlib.h>

#include "utils.h"

void CopyFdToFd(int in_fd, int out_fd) {
  const size_t kBufSize = 0x10000;
  static char buf[kBufSize];
  while (1) {
    ssize_t got = read(in_fd, UNTAG(buf), kBufSize);
    if (got > 0) {
      write(out_fd, UNTAG(buf), got);
    } else if (got == 0) {
      break;
    } else if (errno != EAGAIN || errno != EWOULDBLOCK || errno != EINTR) {
      fprintf(stderr, "error reading file, errno %d\n", errno);
      abort();
    }
  }
}

void *ThreadFn(void *arg) {
  (void)arg;
  int fd = open(UNTAG("/proc/self/maps"), O_RDONLY);
  CopyFdToFd(fd, 2);
  close(fd);
  return NULL;
}

int main(void) {
  pthread_t t;
  void * volatile res = malloc(100);
  void * volatile res2 = malloc(100000);
  pthread_create(&t, 0, ThreadFn, 0);
  pthread_join(t, 0);
  return (int)(size_t)res;
}
