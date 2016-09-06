// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

int fds[2];

void *Thread1(void *x) {
  char buf;
  read(fds[0], &buf, 1);
  close(fds[0]);
  return 0;
}

void *Thread2(void *x) {
  close(fds[1]);
  return 0;
}

int main() {
  fds[0] = open("/dev/random", O_RDONLY);
  fds[1] = dup2(fds[0], 100);
  pthread_t t[2];
  pthread_create(&t[0], NULL, Thread1, NULL);
  pthread_create(&t[1], NULL, Thread2, NULL);
  pthread_join(t[0], NULL);
  pthread_join(t[1], NULL);
  fprintf(stderr, "OK\n");
}

// CHECK-NOT: WARNING: ThreadSanitizer: data race
