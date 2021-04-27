// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
#include "test.h"

void *Thread1(void *x) {
  char buf;
  while (read((int)(long)x, &buf, 1) <= 0) {
  }
  return 0;
}

void *Thread2(void *x) {
  volatile int *stop = (volatile int *)x;
  while (!*stop) {
  }
  return 0;
}

//!!! add create/join threads

int main() {
  int fds[2];
  if (pipe(fds))
    exit((perror("pipe"), 1));
  int stop = 0;
  ANNOTATE_BENIGN_RACE(stop);
  pthread_t t[2];
  pthread_create(&t[0], 0, Thread1, (void *)(long)fds[0]);
  pthread_create(&t[1], 0, Thread2, &stop);
  volatile int x = 0;
  for (int i = 0; i < (1 << 24); i++)
    __atomic_store_n(&x, 1, __ATOMIC_RELEASE);
  if (write(fds[1], fds, 1) < 0)
    exit((perror("write"), 1));
  __atomic_store_n(&stop, 1, __ATOMIC_RELAXED);
  pthread_join(t[0], 0);
  pthread_join(t[1], 0);
  fprintf(stderr, "DONE\n");
  return 0;
}

// CHECK-NOT: ThreadSanitizer:
// CHECK: DONE
// CHECK-NOT: ThreadSanitizer:
