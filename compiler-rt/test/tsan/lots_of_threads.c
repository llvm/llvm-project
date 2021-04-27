// RUN: %clang_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
#include "test.h"

void *thr(void *arg) {
  // Create a sync object on stack, so there is something to free on thread end.
  volatile int x;
  __atomic_fetch_add(&x, 1, __ATOMIC_SEQ_CST);
  barrier_wait(&barrier);
  return 0;
}

int main() {
  const int kThreads = 300;
  barrier_init(&barrier, kThreads + 1);
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setstacksize(&attr, 16 << 20);
  for (int i = 0; i < 10; i++) {
    pthread_t t[kThreads];
    for (int i = 0; i < kThreads; i++)
      pthread_create(&t[i], &attr, thr, 0);
    barrier_wait(&barrier);
    for (int i = 0; i < kThreads; i++)
      pthread_join(t[i], 0);
  }
  pthread_attr_destroy(&attr);
  fprintf(stderr, "DONE\n");
  return 0;
}

// CHECK: DONE

