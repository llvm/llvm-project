// RUN: %clangxx -O0 %s -o %t && %run %t

// pthread_mutexattr_setpshared and pthread_mutexattr_getpshared unavailable
// UNSUPPORTED: netbsd

#include <assert.h>
#include <pthread.h>

int main(void) {
  pthread_mutexattr_t ma;
  int res = pthread_mutexattr_init(&ma);
  assert(res == 0);
  res = pthread_mutexattr_setpshared(&ma, 1);
  assert(res == 0);
  int pshared;
  res = pthread_mutexattr_getpshared(&ma, &pshared);
  assert(res == 0);
  assert(pshared == 1);
  res = pthread_mutexattr_destroy(&ma);
  assert(res == 0);
  return 0;
}
