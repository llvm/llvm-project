// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t 2>&1 | FileCheck %s
#include "test.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

// dup2(oldfd, newfd) races with close(newfd).

int fd;

void *Thread(void *x) {
  barrier_wait(&barrier);
  if (close(fd) == -1)
    exit(printf("close failed\n"));
  return 0;
}

int main() {
  barrier_init(&barrier, 2);
  fd = open("/dev/random", O_RDONLY);
  int fd2 = open("/dev/random", O_RDONLY);
  if (fd == -1 || fd2 == -1)
    exit(printf("open failed\n"));
  pthread_t th;
  pthread_create(&th, 0, Thread, 0);
  if (dup2(fd2, fd) == -1)
    exit(printf("dup2 failed\n"));
  barrier_wait(&barrier);
  pthread_join(th, 0);
  fprintf(stderr, "DONE\n");
}

// CHECK: WARNING: ThreadSanitizer: data race
