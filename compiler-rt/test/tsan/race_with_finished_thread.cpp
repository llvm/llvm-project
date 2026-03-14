// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
#include "test.h"

// Ensure that we can restore a stack of a finished thread.

int g_data;

void __attribute__((noinline)) foobar(int *p) {
  *p = 42;
}

void *Thread1(void *x) {
  foobar(&g_data);
  barrier_wait(&barrier);
  return NULL;
}

void *Thread2(void *x) {
  barrier_wait(&barrier);
  sleep(1); // let the thread finish and exit
  g_data = 43;
  return NULL;
}

int main() {
  barrier_init(&barrier, 2);
  pthread_t t[2];
  pthread_create(&t[0], NULL, Thread1, NULL);
  pthread_create(&t[1], NULL, Thread2, NULL);
  pthread_join(t[0], NULL);
  pthread_join(t[1], NULL);
  return 0;
}

// CHECK: WARNING: ThreadSanitizer: data race
// CHECK:   Write of size 4 at {{.*}} by thread T2:
// CHECK:   Previous write of size 4 at {{.*}} by thread T1:
// CHECK:     #0 foobar
// CHECK:     #1 Thread1
// CHECK:   Thread T1 (tid={{.*}}, finished) created by main thread at:
// CHECK:     #0 pthread_create
// CHECK:     #1 main
