// RUN: %clangxx_tsan -O1 -fno-inline-functions %s -o %t && %deflake %run %t | FileCheck %s
#include "test.h"
#include <memory>

std::unique_ptr<long> global(new long(42));

void *thread(void *x) {
  *global = 43;
  barrier_wait(&barrier);
  return nullptr;
}

int main() {
  barrier_init(&barrier, 2);
  pthread_t th;
  pthread_create(&th, nullptr, thread, nullptr);
  pthread_detach(th);
  barrier_wait(&barrier);
  return 0;
}

// CHECK: WARNING: ThreadSanitizer: data race
// CHECK:   Write of size 8
// The exact spelling and number of std frames is hard to guess.
// CHECK:     unique_ptr
// CHECK:     #{{[1-9]}} cxa_at_exit_callback_installed_at
// CHECK:     #{{[2-9]}} __cxx_global_var_init
