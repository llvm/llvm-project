// RUN: %clangxx_msan -fsanitize-memory-track-origins -O0 %s -o %t && %run %t

// Regression test for a deadlock in pthread_getattr_np

#include <assert.h>
#include <pthread.h>

void *ThreadFn(void *) {
  pthread_attr_t attr;
  int res = pthread_getattr_np(pthread_self(), &attr);
  assert(!res);
  return 0;
}

int main(void) {
  pthread_t t;
  int res = pthread_create(&t, 0, ThreadFn, 0);
  assert(!res);
  res = pthread_join(t, 0);
  assert(!res);
  return 0;
}
