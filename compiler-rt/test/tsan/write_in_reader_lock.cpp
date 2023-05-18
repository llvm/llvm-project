// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
#include "test.h"

pthread_rwlock_t rwlock;
int GLOB;

void *Thread1(void *p) {
  (void)p;
  pthread_rwlock_rdlock(&rwlock);
  barrier_wait(&barrier);
  // Write under reader lock.
  GLOB++;
  pthread_rwlock_unlock(&rwlock);
  return 0;
}

int main(int argc, char *argv[]) {
  barrier_init(&barrier, 2);
  pthread_rwlock_init(&rwlock, NULL);
  pthread_rwlock_rdlock(&rwlock);
  pthread_t t;
  pthread_create(&t, 0, Thread1, 0);
  volatile int x = GLOB;
  (void)x;
  pthread_rwlock_unlock(&rwlock);
  barrier_wait(&barrier);
  pthread_join(t, 0);
  pthread_rwlock_destroy(&rwlock);
  return 0;
}

// CHECK: WARNING: ThreadSanitizer: data race
// CHECK:   Write of size 4 at {{.*}} by thread T1{{.*}}:
// CHECK:     #0 Thread1(void*) {{.*}}write_in_reader_lock.cpp:12
// CHECK:   Previous read of size 4 at {{.*}} by main thread{{.*}}:
// CHECK:     #0 main {{.*}}write_in_reader_lock.cpp:23
