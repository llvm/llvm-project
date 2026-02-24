// RUN: %clangxx_tsan %s -o %t
// RUN: %env_tsan_opts=atexit_sleep_ms=0:abort_on_error=0:simulate_scheduler=random:simulate_iterations=5 %run %t 2>&1 | FileCheck %s

#include <assert.h>
#include <pthread.h>

extern "C" int __tsan_simulate(void (*callback)(void *arg), void *arg);

void *thread_func(void *arg) { return nullptr; }

void test_callback(void *arg) {
  pthread_t t;

  int id1 = 1;
  pthread_create(&t, nullptr, thread_func, &id1);
  int res1 = pthread_join(t, nullptr);
  assert(res1 == 0);

  int id2 = 2;
  pthread_create(&t, nullptr, thread_func, &id2);
  int res2 = pthread_join(t, nullptr);
  assert(res2 == 0);
}

int main() { return __tsan_simulate(test_callback, nullptr); }

// CHECK: simulation starting
