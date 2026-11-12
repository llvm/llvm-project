// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>

struct Cache {
  int x;
  explicit Cache(int x)
    : x(x) {
  }
};

void foo(Cache *my) {
  static Cache *c = my ? my : new Cache(rand());
  if (c->x >= RAND_MAX)
    exit(1);
}

void *Thread(void *x) {
  foo(new Cache(rand()));
  return 0;
}

int main() {
  pthread_t t[2];
  pthread_create(&t[0], 0, Thread, 0);
  pthread_create(&t[1], 0, Thread, 0);
  pthread_join(t[0], 0);
  pthread_join(t[1], 0);
  fprintf(stderr, "PASS\n");
}

// CHECK-NOT: WARNING: ThreadSanitizer: data race
