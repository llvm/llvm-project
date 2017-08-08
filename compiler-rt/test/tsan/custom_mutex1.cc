// RUN: %clangxx_tsan -O1 --std=c++11 %s -o %t && %deflake %run %t 2>&1 | FileCheck %s
#include "custom_mutex.h"

// Test that failed TryLock does not induce parasitic synchronization.

Mutex mu(true, 0);
long data;

void *thr(void *arg) {
  mu.Lock();
  data++;
  mu.Unlock();
  mu.Lock();
  barrier_wait(&barrier);
  barrier_wait(&barrier);
  mu.Unlock();
  return 0;
}

int main() {
  barrier_init(&barrier, 2);
  pthread_t th;
  pthread_create(&th, 0, thr, 0);
  barrier_wait(&barrier);
  if (mu.TryLock()) {
    fprintf(stderr, "TryLock succeeded, should not\n");
    exit(0);
  }
  data++;
  barrier_wait(&barrier);
  pthread_join(th, 0);
  fprintf(stderr, "DONE\n");
  return 0;
}

// CHECK: ThreadSanitizer: data race
// CHECK-NEXT:   Write of size 8 at {{.*}} by main thread:
// CHECK-NEXT:     #0 main {{.*}}custom_mutex1.cc:29
// CHECK: DONE
