// RUN: %clang_scudo %s -o %t
// RUN: %run %t 2>&1

#include <locale.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// Some of glibc's own thread local data is destroyed after a user's thread
// local destructors are called, via __libc_thread_freeres. This might involve
// calling free, as is the case for strerror_thread_freeres.
// If there is no prior heap operation in the thread, this free would end up
// initializing some thread specific data that would never be destroyed
// properly, while still being deallocated when the TLS goes away. As a result,
// a program could SEGV, usually in
// __sanitizer::AllocatorGlobalStats::Unregister, where one of the doubly
// linked list links would refer to a now unmapped memory area.

// This test reproduces those circumstances. Success means executing without
// a segmentation fault.

const int kNumThreads = 16;
pthread_t tid[kNumThreads];

void *thread_func(void *arg) {
  uintptr_t i = (uintptr_t)arg;
  if ((i & 1) == 0)
    free(malloc(16));
  // Calling strerror_l allows for strerror_thread_freeres to be called.
  strerror_l(0, LC_GLOBAL_LOCALE);
  return 0;
}

int main(int argc, char **argv) {
  for (uintptr_t j = 0; j < 8; j++) {
    for (uintptr_t i = 0; i < kNumThreads; i++)
      pthread_create(&tid[i], 0, thread_func, (void *)i);
    for (uintptr_t i = 0; i < kNumThreads; i++)
      pthread_join(tid[i], 0);
  }
  return 0;
}
