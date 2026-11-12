// RUN: %clangxx_msan -O0 %s -o %t && %run %t

// Check that when TLS block is reused between threads, its shadow is cleaned.

#include <pthread.h>
#include <stdio.h>

int __thread x;

void *ThreadFn(void *) {
  if (!x)
    printf("zzz\n");
  int y;
  int * volatile p = &y;
  x = *p;
  return 0;
}

int main(void) {
  pthread_t t;
  for (int i = 0; i < 100; ++i) {
    pthread_create(&t, 0, ThreadFn, 0);
    pthread_join(t, 0);
  }
  return 0;
}
