// RUN: %clangxx_tsan -O1 %s -o %t && %env_tsan_opts=print_full_thread_history=true %deflake %run %t 2>&1 | FileCheck %s

#include "test.h"

int Global;

void *Thread2(void *x) {
  barrier_wait(&barrier);
  Global++;
  return NULL;
}

void *Thread3(void *x) {
  Global--;
  barrier_wait(&barrier);
  return NULL;
}

void *Thread1(void *x) {
  pthread_t t[2];
  pthread_create(&t[0], NULL, Thread2, NULL);
  pthread_create(&t[1], NULL, Thread3, NULL);
  pthread_join(t[0], NULL);
  pthread_join(t[1], NULL);
  return NULL;
}

int main() {
  barrier_init(&barrier, 2);
  pthread_t t;
  pthread_create(&t, NULL, Thread1, NULL);
  pthread_join(t, NULL);
  return 0;
}

// CHECK: WARNING: ThreadSanitizer: data race
// CHECK:        Thread T2 {{.*}} created by thread T1 at
// CHECK:        Thread T3 {{.*}} created by thread T1 at:
// CHECK:        Thread T1 {{.*}} created by main thread at:
// CHECK: SUMMARY: ThreadSanitizer: data race{{.*}}
