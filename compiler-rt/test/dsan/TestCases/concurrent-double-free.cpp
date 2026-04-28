// REQUIRES: linux
// RUN: %clangxx_dsan %s -pthread -o %t
// RUN: not %run %t 2>&1 | FileCheck %s

#include <pthread.h>
#include <stdlib.h>

static pthread_barrier_t barrier;
static void *p;

static void *Free(void *) {
  pthread_barrier_wait(&barrier);
  free(p);
  return nullptr;
}

int main() {
  pthread_t first, second;
  p = malloc(16);
  pthread_barrier_init(&barrier, nullptr, 2);
  pthread_create(&first, nullptr, Free, nullptr);
  pthread_create(&second, nullptr, Free, nullptr);
  pthread_join(first, nullptr);
  pthread_join(second, nullptr);
  return 0;
}

// CHECK: ERROR: DoubleFreeSanitizer: double-free on address
