// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
#include "test.h"

int fds[2];

void *ThreadCreatePipe(void *x) {
  pipe(fds);
  return NULL;
}

void *ThreadDummy(void *x) {
  return NULL;
}

void *ThreadWrite(void *x) {
  write(fds[1], "a", 1);
  barrier_wait(&barrier);
  return NULL;
}

void *ThreadClose(void *x) {
  barrier_wait(&barrier);
  close(fds[0]);
  close(fds[1]);
  return NULL;
}

int main() {
  barrier_init(&barrier, 2);
  pthread_t t_create;
  pthread_create(&t_create, NULL, ThreadCreatePipe, NULL);
  pthread_join(t_create, NULL);

  for (int i = 0; i < 100; i++) {
    pthread_t t_dummy;
    pthread_create(&t_dummy, NULL, ThreadDummy, NULL);
    pthread_join(t_dummy, NULL);
  }

  pthread_t t[2];
  pthread_create(&t[0], NULL, ThreadWrite, NULL);
  pthread_create(&t[1], NULL, ThreadClose, NULL);
  pthread_join(t[0], NULL);
  pthread_join(t[1], NULL);
}

// CHECK-NOT: CHECK failed
// CHECK: WARNING: ThreadSanitizer: data race
// CHECK:   Write of size 8
// CHECK:     #0 close
// CHECK:     #1 ThreadClose
// CHECK:   Previous read of size 8
// CHECK:     #0 write
// CHECK:     #1 ThreadWrite
