// RUN: %clang_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s

// pthread barriers are not available on OS X
// UNSUPPORTED: darwin

#include <pthread.h>
#include <stdio.h>
#include <stddef.h>
#include <unistd.h>

pthread_barrier_t B;
int Global;

void *Thread1(void *x) {
  if (pthread_barrier_wait(&B) == PTHREAD_BARRIER_SERIAL_THREAD)
    pthread_barrier_destroy(&B);
  return NULL;
}

void *Thread2(void *x) {
  if (pthread_barrier_wait(&B) == PTHREAD_BARRIER_SERIAL_THREAD)
    pthread_barrier_destroy(&B);
  return NULL;
}

int main() {
  pthread_barrier_init(&B, 0, 2);
  pthread_t t;
  pthread_create(&t, NULL, Thread1, NULL);
  Thread2(0);
  pthread_join(t, NULL);
  return 0;
}

// CHECK:      WARNING: ThreadSanitizer: data race
