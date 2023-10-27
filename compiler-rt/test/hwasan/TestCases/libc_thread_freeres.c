// RUN: %clang_hwasan %s -o %t && %env_hwasan_opts=random_tags=1 %run %t

#include <pthread.h>
#include <sanitizer/hwasan_interface.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void *ThreadFn(void *) {
  strerror_l(-1, 0);
  __hwasan_enable_allocator_tagging();
  // This will trigger memory deallocation in __strerror_thread_freeres,
  // at a point when HwasanThread is already gone.
}

int main() {
  pthread_t t;
  pthread_create(&t, NULL, ThreadFn, NULL);
  pthread_join(t, NULL);
  return 0;
}
