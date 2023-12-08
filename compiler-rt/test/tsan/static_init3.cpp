// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <sched.h>

struct Cache {
  int x;
};

Cache g_cache;

Cache *CreateCache() {
  g_cache.x = rand();
  return &g_cache;
}

_Atomic(Cache*) queue;

void *Thread1(void *x) {
  static Cache *c = CreateCache();
  __c11_atomic_store(&queue, c, 0);
  return 0;
}

void *Thread2(void *x) {
  Cache *c = 0;
  for (;;) {
    c = __c11_atomic_load(&queue, 0);
    if (c)
      break;
    sched_yield();
  }
  if (c->x >= RAND_MAX)
    exit(1);
  return 0;
}

int main() {
  pthread_t t[2];
  pthread_create(&t[0], 0, Thread1, 0);
  pthread_create(&t[1], 0, Thread2, 0);
  pthread_join(t[0], 0);
  pthread_join(t[1], 0);
}

// CHECK: WARNING: ThreadSanitizer: data race
