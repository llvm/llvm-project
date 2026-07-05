// Sanitizer should not crash if pthread_create fails.
// RUN: %clangxx -pthread %s -o %t && %run %t

// pthread_create with lsan i386 does not fail here.
// UNSUPPORTED: i386-linux && lsan

#include <cassert>
#include <pthread.h>
#include <stdlib.h>

void *null_func(void *args) {
  return NULL;
}

int main(void) {
  pthread_t thread;
  pthread_attr_t attrs;
  pthread_attr_init(&attrs);
  // Set size huge enough to fail pthread_create.
  size_t sz = ~0;
  // Align the size just in case.
  sz >>= 16;
  sz <<= 16;
  int res = pthread_attr_setstacksize(&attrs, sz);
  assert(res == 0);
  for (size_t i = 0; i < 10; ++i) {
    res = pthread_create(&thread, &attrs, null_func, NULL);
    assert(res != 0);
  }
  return 0;
}
