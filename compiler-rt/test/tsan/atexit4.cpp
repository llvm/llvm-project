// RUN: %clang_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
#include "test.h"

void *thread(void *x) {
  barrier_wait(&barrier);
  *static_cast<int *>(x) = 2;
  return nullptr;
}

static void race() {
  int data = 0;
  pthread_t t;
  pthread_create(&t, nullptr, thread, &data);
  data = 1;
  barrier_wait(&barrier);
  pthread_join(t, nullptr);
}

struct X {
  __attribute__((noinline))
  X() { atexit(race); }
} x;

int main() {
  barrier_init(&barrier, 2);
  fprintf(stderr, "DONE\n");
}

// CHECK: DONE
// CHECK: WARNING: ThreadSanitizer: data race
// CHECK:   Write of size 4
// CHECK:     #0 thread
// CHECK:   Previous write of size 4
// CHECK:     #0 race
// CHECK:     #1 at_exit_callback_installed_at
// CHECK:     #2 X
