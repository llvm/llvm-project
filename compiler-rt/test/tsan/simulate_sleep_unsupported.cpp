// RUN: %clangxx_tsan -O1 %s -o %t && env TSAN_OPTIONS="simulate_scheduler=random:simulate_iterations=10" not %run %t 2>&1 | FileCheck %s

#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>

extern "C" int __tsan_simulate(void (*callback)(void *arg), void *arg);

void *thread_func(void *arg) {
  usleep(1000);
  return nullptr;
}

void test_callback(void *arg) {
  pthread_t t;
  pthread_create(&t, nullptr, thread_func, nullptr);
  pthread_join(t, nullptr);
}

int main() { return __tsan_simulate(test_callback, nullptr); }

// CHECK: ThreadSanitizer: simulation error - unsupported interceptor called: usleep
// CHECK: Simulation does not support this synchronization primitive
// CHECK: ThreadSanitizer: simulation aborted after 1 iterations
