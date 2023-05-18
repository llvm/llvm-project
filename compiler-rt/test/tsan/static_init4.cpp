// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <sched.h>

struct Cache {
  int x;
  explicit Cache(int x)
    : x(x) {
  }
};

int g_other;

Cache *CreateCache() {
  g_other = rand();
  return new Cache(rand());
}

void *Thread1(void *x) {
  static Cache *c = CreateCache();
  if (c->x == g_other)
    exit(1);
  return 0;
}

int main() {
  pthread_t t[2];
  pthread_create(&t[0], 0, Thread1, 0);
  pthread_create(&t[1], 0, Thread1, 0);
  pthread_join(t[0], 0);
  pthread_join(t[1], 0);
  fprintf(stderr, "PASS\n");
}

// CHECK-NOT: WARNING: ThreadSanitizer: data race
