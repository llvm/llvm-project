// RUN: %clangxx_tsan -O1 --std=c++11 %s -o %t && %deflake %run %t 2>&1 | FileCheck %s
#include "custom_mutex.h"

// Test that Broadcast does not induce parasitic synchronization.

Mutex mu;
long data;

void *thr(void *arg) {
  barrier_wait(&barrier);
  mu.Lock();
  data++;
  mu.Unlock();
  data++;
  mu.Broadcast();
  return 0;
}

int main() {
  barrier_init(&barrier, 2);
  pthread_t th;
  pthread_create(&th, 0, thr, 0);
  mu.Lock();
  barrier_wait(&barrier);
  while (data == 0)
    mu.Wait();
  mu.Unlock();
  pthread_join(th, 0);
  fprintf(stderr, "DONE\n");
  return 0;
}

// CHECK: ThreadSanitizer: data race
// CHECK: DONE
