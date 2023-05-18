// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
#include "test.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>

int fd;
char buf;

void *Thread1(void *x) {
  barrier_wait(&barrier);
  read(fd, &buf, 1);
  return NULL;
}

void *Thread2(void *x) {
  read(fd, &buf, 1);
  barrier_wait(&barrier);
  return NULL;
}

int main() {
  barrier_init(&barrier, 2);
  fd = open("/dev/random", O_RDONLY);
  if (fd < 0) {
    fprintf(stderr, "failed to open /dev/random (%d)\n", errno);
    return 1;
  }
  pthread_t t[2];
  pthread_create(&t[0], NULL, Thread1, NULL);
  pthread_create(&t[1], NULL, Thread2, NULL);
  pthread_join(t[0], NULL);
  pthread_join(t[1], NULL);
  close(fd);
  fprintf(stderr, "DONE\n");
}

// CHECK: WARNING: ThreadSanitizer: data race
// CHECK:   Write of size 1
// CHECK:     #0 read
// CHECK:   Previous write of size 1
// CHECK:     #0 read
// CHECK: DONE

