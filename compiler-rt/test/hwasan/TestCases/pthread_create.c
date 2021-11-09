// Tests that our thread initialization hooks work properly with random_tags=1.
// RUN: %clang_hwasan %s -o %t
// RUN: %env_hwasan_opts=random_tags=1 %run %t
// REQUIRES: stable-runtime

#include <pthread.h>

#include <sanitizer/hwasan_interface.h>

volatile int state;

void *Increment(void *arg) {
  ++state;
  return NULL;
}

int main() {
  __hwasan_enable_allocator_tagging();
  pthread_t t1;
  pthread_create(&t1, NULL, Increment, NULL);
  pthread_join(t1, NULL);
}
