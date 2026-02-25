// RUN: %clangxx_tsan -O1 %s -fsanitize-thread-simulate-main -o %t
// RUN: %env_tsan_opts=atexit_sleep_ms=0:abort_on_error=0:simulate_scheduler=random:simulate_iterations=50 not %run %t 2>&1 | FileCheck %s
//
// REQUIRES: linux

#include <atomic>
#include <pthread.h>

std::atomic<int> d{};
int a = 0;

void *thread_func(void *arg) {
  ++d;
  ++a; // Data race!
  ++d;
  return nullptr;
}

// Note: NO call to __tsan_simulate() - the -fsanitize-thread-simulate-main
// flag automatically wraps this main() to run under simulation.
int main() {
  pthread_t t1, t2;
  pthread_create(&t1, nullptr, thread_func, nullptr);
  pthread_create(&t2, nullptr, thread_func, nullptr);
  pthread_join(t1, nullptr);
  pthread_join(t2, nullptr);
  return 0;
}

// CHECK: ThreadSanitizer: simulation starting
// CHECK: WARNING: ThreadSanitizer: data race
// CHECK: Write of size 4
// CHECK: Previous write of size 4
// CHECK: ThreadSanitizer: data race detected at iteration
