// RUN: %clang_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s

// on_exit() is not available on Darwin.
// UNSUPPORTED: darwin

#include "test.h"

volatile long global;

void *thread(void *x) {
  global++;
  barrier_wait(&barrier);
  return nullptr;
}

void on_exit_callback(int status, void *arg) {
  fprintf(stderr, "on_exit_callback(%d, %lu)\n", status, (long)arg);
  global++;
}

int main() {
  on_exit(on_exit_callback, (void *)42l);
  barrier_init(&barrier, 2);
  pthread_t th;
  pthread_create(&th, nullptr, thread, nullptr);
  pthread_detach(th);
  barrier_wait(&barrier);
  return 2;
}

// CHECK: on_exit_callback(2, 42)
// CHECK: WARNING: ThreadSanitizer: data race
// CHECK:   Write of size 8
// CHECK:     #0 on_exit_callback
// CHECK:     #1 on_exit_callback_installed_at
// CHECK:     #2 main
