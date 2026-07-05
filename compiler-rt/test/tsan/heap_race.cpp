// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
#include "test.h"
#include <pthread.h>
#include <stdio.h>
#include <stddef.h>

void *Thread(void *a) {
  ((int*)a)[0]++;
  barrier_wait(&barrier);
  return NULL;
}

int main() {
  barrier_init(&barrier, 2);
  int *p = new int(42);
  pthread_t t;
  pthread_create(&t, NULL, Thread, p);
  barrier_wait(&barrier);
  p[0]++;
  pthread_join(t, NULL);
  delete p;
}

// CHECK: WARNING: ThreadSanitizer: data race
