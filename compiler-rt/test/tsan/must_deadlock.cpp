// Test that the deadlock detector can find a deadlock that actually happened.
// Currently we will fail to report such a deadlock because we check for
// cycles in lock-order graph after pthread_mutex_lock.

// RUN: %clangxx_tsan %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s
// XFAIL: *
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

pthread_mutex_t mu1, mu2;
pthread_barrier_t barrier;

void *Thread(void *p) {
  // mu2 => mu1
  pthread_mutex_lock(&mu2);
  pthread_barrier_wait(&barrier);
  pthread_mutex_lock(&mu1);
  // CHECK: ThreadSanitizer: lock-order-inversion (potential deadlock)
  pthread_mutex_unlock(&mu1);
  pthread_mutex_unlock(&mu2);
  return p;
}

int main() {
  pthread_mutex_init(&mu1, NULL);
  pthread_mutex_init(&mu2, NULL);
  pthread_barrier_init(&barrier, 0, 2);

  fprintf(stderr, "This test is going to deadlock and die in 3 seconds\n");
  alarm(3);

  pthread_t t;
  pthread_create(&t, 0, Thread, 0);

  // mu1 => mu2
  pthread_mutex_lock(&mu1);
  pthread_barrier_wait(&barrier);
  pthread_mutex_lock(&mu2);
  pthread_mutex_unlock(&mu2);
  pthread_mutex_unlock(&mu1);

  pthread_join(t, 0);

  pthread_mutex_destroy(&mu1);
  pthread_mutex_destroy(&mu2);
  pthread_barrier_destroy(&barrier);
  fprintf(stderr, "FAILED\n");
}
