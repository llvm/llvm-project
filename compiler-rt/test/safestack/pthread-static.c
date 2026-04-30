// RUN: %clang_safestack %s -pthread -static -o %t
// RUN: %run %t

// REQUIRES: linux || freebsd

// Smoke test for pthread_create in a statically linked executable.

#include <pthread.h>

void *f(void *p) {
  return NULL;
}

int main(void) {
  pthread_t t;
  (void)pthread_create(&t, NULL, f, NULL);
  return 0;
}
