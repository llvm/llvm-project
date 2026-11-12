// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
#include "test.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

int fd;
char buf;

void *Thread1(void *x) {
  buf = 1;
  barrier_wait(&barrier);
  return NULL;
}

void *Thread2(void *x) {
  write(fd, &buf, 1);
  return NULL;
}

int main() {
  barrier_init(&barrier, 2);
  fd = open("/dev/null", O_WRONLY);
  if (fd < 0) return 1;
  pthread_t t[2];
  pthread_create(&t[0], NULL, Thread1, NULL);
  barrier_wait(&barrier);
  pthread_create(&t[1], NULL, Thread2, NULL);
  pthread_join(t[0], NULL);
  pthread_join(t[1], NULL);
  close(fd);
}

// CHECK: WARNING: ThreadSanitizer: data race
// CHECK:   Read of size 1
// CHECK:     #0 write
// CHECK:   Previous write of size 1
// CHECK:     #0 Thread1
