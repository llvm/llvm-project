// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>

struct P {
  int x;
  int y;
};

void *Thread(void *x) {
  static P p = {rand(), rand()};
  if (p.x > RAND_MAX || p.y > RAND_MAX)
    exit(1);
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
