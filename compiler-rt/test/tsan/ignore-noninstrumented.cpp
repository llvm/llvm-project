// Check that ignore_noninstrumented_modules=1 suppresses reports originating
// from interceptors that are called from an un-instrumented library.

// RUN: %clangxx_tsan %s -fPIC -shared -DLIBRARY -fno-sanitize=thread -o %t.library.so
// RUN: %clangxx_tsan %s %t.library.so -o %t

// Check that without the flag, there are false positives.
// RUN: %env_tsan_opts=ignore_noninstrumented_modules=0 %deflake %run %t      2>&1 | FileCheck %s --check-prefix=CHECK-RACE

// With ignore_noninstrumented_modules=1, no races are reported.
// RUN: %env_tsan_opts=ignore_noninstrumented_modules=1          %run %t      2>&1 | FileCheck %s --implicit-check-not='ThreadSanitizer'

// With ignore_noninstrumented_modules=1, races in user's code are still reported.
// RUN: %env_tsan_opts=ignore_noninstrumented_modules=1 %deflake %run %t race 2>&1 | FileCheck %s --check-prefix=CHECK-RACE

#include "test.h"

#include <cstring>

#ifdef LIBRARY
namespace library {
#endif
char global_buf[64];

void *Thread1(void *arg) {
  auto barrier_wait = (void (*)())arg;
  barrier_wait();
  strcpy(global_buf, "hello world");  // NOLINT
  return NULL;
}

void *Thread2(void *arg) {
  auto barrier_wait = (void (*)())arg;
  strcpy(global_buf, "world hello");  // NOLINT
  barrier_wait();
  return NULL;
}

void Race(void (*barrier_wait)()) {
  pthread_t t[2];
  pthread_create(&t[0], NULL, Thread1, (void *)barrier_wait);
  pthread_create(&t[1], NULL, Thread2, (void *)barrier_wait);
  pthread_join(t[0], NULL);
  pthread_join(t[1], NULL);
}
#ifdef LIBRARY
} // namespace library
#endif

#ifndef LIBRARY
namespace library {
  void Race(void (*barrier_wait)());
}

// Pass pointer to this function to un-instrumented library, so it can access
// TSan-invisible barriers.
void my_barrier_wait() {
  barrier_wait(&barrier);
}

int main(int argc, char *argv[]) {
  fprintf(stderr, "Hello world.\n");

  // Race in un-instrumented library
  barrier_init(&barrier, 2);
  library::Race(my_barrier_wait);

  // Race in user code, if requested
  if (argc > 1 && strcmp(argv[1], "race") == 0) {
    barrier_init(&barrier, 2);
    Race(my_barrier_wait);
  }

  fprintf(stderr, "Done.\n");
}

#endif // LIBRARY

// CHECK: Hello world.
// CHECK-RACE: SUMMARY: ThreadSanitizer: data race
// CHECK: Done.
