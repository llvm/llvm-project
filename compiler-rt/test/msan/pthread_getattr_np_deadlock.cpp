// RUN: %clangxx_msan -fsanitize-memory-track-origins -O0 %s -o %t && %run %t

// Regression test for a deadlock in pthread_getattr_np

#include <assert.h>
#include <pthread.h>
#if defined(__FreeBSD__)
#include <pthread_np.h>
#endif

void *ThreadFn(void *) {
  pthread_attr_t attr;
#if defined(__FreeBSD__)
  // On FreeBSD it needs to allocate attr underlying memory
  int res = pthread_attr_init(&attr);
  assert(!res);
  res = pthread_attr_get_np(pthread_self(), &attr);
#else
  int res = pthread_getattr_np(pthread_self(), &attr);
#endif
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
