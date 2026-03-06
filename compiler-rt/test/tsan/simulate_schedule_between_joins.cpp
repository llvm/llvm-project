// RUN: %clangxx_tsan -O1 %s -o %t
// RUN: %env_tsan_opts=atexit_sleep_ms=0:abort_on_error=0:simulate_scheduler=random:simulate_iterations=50 %run %t 2>&1 | FileCheck %s

#include <atomic>
#include <pthread.h>

extern "C" int __tsan_simulate(void (*callback)(void *), void *arg);

void *thread_func(void *arg) { return nullptr; }

void test_callback(void *arg) {
  pthread_t t1, t2;
  pthread_create(&t1, nullptr, thread_func, nullptr);
  pthread_create(&t2, nullptr, thread_func, nullptr);
  pthread_join(t1, nullptr);

  // Verify simulation scheduling between joins does allow two threads to run
  // in parallel (checked by internal assertions). Only one thread can ever
  // run at the same time in the simulation scheduler.
  std::atomic<int> a{};
  ++a;

  pthread_join(t2, nullptr);
}

int main() { return __tsan_simulate(test_callback, nullptr); }

// CHECK: ThreadSanitizer: simulation starting
