// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
#include "test.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

int X;

void *Thread1(void *x) {
  barrier_wait(&barrier);
  int f = open("/dev/random", O_RDONLY);
  char buf;
  read(f, &buf, 1);
  close(f);
  X = 42;
  return NULL;
}

void *Thread2(void *x) {
  X = 43;
  write(STDOUT_FILENO, "a", 1);
  barrier_wait(&barrier);
  return NULL;
}

int main() {
  barrier_init(&barrier, 2);
  pthread_t t[2];
  pthread_create(&t[0], NULL, Thread1, NULL);
  pthread_create(&t[1], NULL, Thread2, NULL);
  pthread_join(t[0], NULL);
  pthread_join(t[1], NULL);
}

// CHECK: WARNING: ThreadSanitizer: data race
// CHECK:   Write of size 4
// CHECK:     #0 Thread1
// CHECK:   Previous write of size 4
// CHECK:     #0 Thread2


