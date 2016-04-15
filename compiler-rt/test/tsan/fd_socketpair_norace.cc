// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>

int fds[2];
int X;

void *Thread1(void *x) {
  X = 42;
  write(fds[1], "a", 1);
  close(fds[1]);
  return NULL;
}

void *Thread2(void *x) {
  char buf;
  while (read(fds[0], &buf, 1) != 1) {
  }
  X = 43;
  close(fds[0]);
  return NULL;
}

int main() {
  socketpair(AF_UNIX, SOCK_STREAM, 0, fds);
  pthread_t t[2];
  pthread_create(&t[0], NULL, Thread1, NULL);
  pthread_create(&t[1], NULL, Thread2, NULL);
  pthread_join(t[0], NULL);
  pthread_join(t[1], NULL);
  fprintf(stderr, "OK\n");
}

// CHECK-NOT: WARNING: ThreadSanitizer: data race
