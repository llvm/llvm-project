// RUN: %clangxx_msan -std=c++11 -O0 %s -o %t && %run %t
// UNSUPPORTED: android, netbsd

// Regression test for a deadlock in pthread_getattr_np

#include <assert.h>
#include <pthread.h>
#include <string.h>
#include <sanitizer/msan_interface.h>

#include <stdio.h>

void *ThreadFn(void *) {
  return nullptr;
}

int main(void) {
  pthread_t t;
  int res = pthread_create(&t, 0, ThreadFn, 0);
  assert(!res);

  const char *kMyThreadName = "my-thread-name";
  res = pthread_setname_np(t, kMyThreadName);
  assert(!res);

  char buf[100];
  res = pthread_getname_np(t, buf, sizeof(buf));
  assert(!res);
  assert(strcmp(buf, kMyThreadName) == 0);

  res = pthread_join(t, 0);
  assert(!res);
  return 0;
}
