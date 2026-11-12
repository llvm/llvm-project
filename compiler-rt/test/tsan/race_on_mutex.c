// RUN: %clang_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
#include "test.h"

pthread_mutex_t Mtx;
int Global;

void *Thread2(void *x) {
  barrier_wait(&barrier);
// CHECK:      WARNING: ThreadSanitizer: data race
// CHECK-NEXT:   Atomic read of size 1 at {{.*}} by thread T2:
// CHECK-NEXT:     #0 pthread_mutex_lock
// CHECK-NEXT:     #1 Thread2{{.*}} {{.*}}race_on_mutex.c:[[@LINE+1]]{{(:3)?}} ({{.*}})
  pthread_mutex_lock(&Mtx);
  Global = 43;
  pthread_mutex_unlock(&Mtx);
  return NULL;
}

void *Thread1(void *x) {
// CHECK:        Previous write of size {{[0-9]+}} at {{.*}} by thread T1:
// CHECK:          #{{[0-9]+}} {{.*}}pthread_mutex_init {{.*}} ({{.*}})
// CHECK-NEXT:     #{{[0-9]+}} Thread1{{.*}} {{.*}}race_on_mutex.c:[[@LINE+1]]{{(:3)?}} ({{.*}})
  pthread_mutex_init(&Mtx, 0);
  pthread_mutex_lock(&Mtx);
  Global = 42;
  pthread_mutex_unlock(&Mtx);
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
  pthread_mutex_destroy(&Mtx);
  return 0;
}
