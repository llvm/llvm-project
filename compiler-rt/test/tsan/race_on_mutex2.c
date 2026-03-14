// RUN: %clang_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
#include "test.h"

void *Thread(void *x) {
  pthread_mutex_lock((pthread_mutex_t*)x);
  pthread_mutex_unlock((pthread_mutex_t*)x);
  barrier_wait(&barrier);
  return 0;
}

int main() {
  barrier_init(&barrier, 2);
  pthread_mutex_t Mtx;
  pthread_mutex_init(&Mtx, 0);
  pthread_t t;
  pthread_create(&t, 0, Thread, &Mtx);
  barrier_wait(&barrier);
  pthread_mutex_destroy(&Mtx);
  pthread_join(t, 0);
  return 0;
}

// CHECK:      WARNING: ThreadSanitizer: data race
