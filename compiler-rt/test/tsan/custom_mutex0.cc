// RUN: %clangxx_tsan -O1 --std=c++11 %s -o %t && %run %t 2>&1 | FileCheck %s
#include "custom_mutex.h"

// Test that custom annoations provide normal mutex synchronization
// (no race reports for properly protected critical sections).

Mutex mu(true, 0);
long data;

void *thr(void *arg) {
  barrier_wait(&barrier);
  mu.Lock();
  data++;
  mu.Unlock();
  return 0;
}

int main() {
  barrier_init(&barrier, 2);
  pthread_t th;
  pthread_create(&th, 0, thr, 0);
  barrier_wait(&barrier);
  mu.Lock();
  data++;
  mu.Unlock();
  pthread_join(th, 0);
  fprintf(stderr, "DONE\n");
  return 0;
}

// CHECK: DONE
