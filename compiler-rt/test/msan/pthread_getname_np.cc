// RUN: %clangxx_msan -std=c++11 -O0 %s -o %t && %run %t
// UNSUPPORTED: android, netbsd

// Regression test for a deadlock in pthread_getattr_np

#include <assert.h>
#include <pthread.h>
#include <string.h>
#include <sanitizer/msan_interface.h>

#include <stdio.h>

// Stall child thread on this lock to make sure it doesn't finish
// before the end of the pthread_getname_np() / pthread_setname_np() tests.
static pthread_mutex_t lock;

void *ThreadFn(void *) {
  pthread_mutex_lock (&lock);
  pthread_mutex_unlock (&lock);
  return nullptr;
}

int main(void) {
  pthread_t t;

  pthread_mutex_init (&lock, NULL);
  pthread_mutex_lock (&lock);

  int res = pthread_create(&t, 0, ThreadFn, 0);
  assert(!res);

  const char *kMyThreadName = "my-thread-name";
  res = pthread_setname_np(t, kMyThreadName);
  assert(!res);

  char buf[100];
  res = pthread_getname_np(t, buf, sizeof(buf));
  assert(!res);
  assert(strcmp(buf, kMyThreadName) == 0);

  pthread_mutex_unlock (&lock);

  res = pthread_join(t, 0);
  assert(!res);
  return 0;
}
