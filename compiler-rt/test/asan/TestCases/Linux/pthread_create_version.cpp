// RUN: %clangxx_asan -std=c++11 -pthread %s -o %t && %run %t 2>&1
// Regression test for the versioned pthread_create interceptor on linux/i386.
// pthread_attr_init is not intercepted and binds to the new abi
// pthread_create is intercepted; dlsym always returns the oldest version.
// This results in a crash inside pthread_create in libc.

#include <pthread.h>
#include <stdlib.h>

void *ThreadFunc(void *) { return nullptr; }

int main() {
  pthread_t t;
  const size_t sz = 1024 * 1024;
  void *p = malloc(sz);
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setstack(&attr, p, sz);
  pthread_create(&t, &attr, ThreadFunc, nullptr);
  pthread_join(t, nullptr);
  free(p);
  return 0;
}
