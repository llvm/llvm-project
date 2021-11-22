// RUN: %clangxx_tsan %darwin_min_target_with_tls_support -O1 %s -o %t && \
// RUN:   %deflake %run %t | FileCheck %s
#include "test.h"

int Global;
__thread int huge[1024*1024];

void *Thread1(void *x) {
  barrier_wait(&barrier);
  Global++;
  return NULL;
}

void *Thread2(void *x) {
  pthread_mutex_t *mtx = new pthread_mutex_t;
  pthread_mutex_init(mtx, 0);
  pthread_mutex_lock(mtx);
  Global--;
  pthread_mutex_unlock(mtx);
  pthread_mutex_destroy(mtx);
  delete mtx;
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
// CHECK: Write of size 4 at {{.*}} by thread T1:
// CHECK: Previous write of size 4 at {{.*}} by thread T2
// CHECK:                                      (mutexes: write [[M1:M[0-9]+]]):
// CHECK: Mutex [[M1]] is already destroyed
// CHECK-NOT: Mutex {{.*}} created at

