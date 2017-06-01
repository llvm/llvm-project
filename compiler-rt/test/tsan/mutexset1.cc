// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
#include "test.h"

int Global;
pthread_mutex_t mtx;

void *Thread1(void *x) {
  barrier_wait(&barrier);
  pthread_mutex_lock(&mtx);
  Global++;
  pthread_mutex_unlock(&mtx);
  return NULL;
}

void *Thread2(void *x) {
  Global--;
  barrier_wait(&barrier);
  return NULL;
}

int main() {
  barrier_init(&barrier, 2);
  // CHECK: WARNING: ThreadSanitizer: data race
  // CHECK:   Write of size 4 at {{.*}} by thread T1
  // CHECK:                         (mutexes: write [[M1:M[0-9]+]]):
  // CHECK:   Previous write of size 4 at {{.*}} by thread T2:
  // CHECK:   Mutex [[M1]] (0x{{.*}}) created at:
  // CHECK:     #0 pthread_mutex_init
  // CHECK:     #1 main {{.*}}mutexset1.cc:[[@LINE+1]]
  pthread_mutex_init(&mtx, 0);
  pthread_t t[2];
  pthread_create(&t[0], NULL, Thread1, NULL);
  pthread_create(&t[1], NULL, Thread2, NULL);
  pthread_join(t[0], NULL);
  pthread_join(t[1], NULL);
  pthread_mutex_destroy(&mtx);
}
