// RUN: %clangxx_tsan -O1 %s -o %t
// RUN: %env_tsan_opts=atexit_sleep_ms=0:abort_on_error=0:simulate_scheduler=random:simulate_iterations=100:simulate_max_depth=100 not %run %t 2>&1 | FileCheck %s

#include <assert.h>
#include <atomic>
#include <pthread.h>

extern "C" int __tsan_simulate(void (*callback)(void *), void *arg);

static std::atomic<int> counter(0);

void *thread_func(void *arg) {
  for (int i = 0; i < 200; i++) {
    counter.fetch_add(1, std::memory_order_relaxed);
  }
  return nullptr;
}

static int called;

void test_callback(void *arg) {
  counter.store(0, std::memory_order_relaxed);

  pthread_t t1, t2;
  pthread_create(&t1, nullptr, thread_func, nullptr);
  pthread_create(&t2, nullptr, thread_func, nullptr);

  pthread_join(t1, nullptr);
  pthread_join(t2, nullptr);

  ++called;
}

int main() {
  called = 0;
  int result = __tsan_simulate(test_callback, nullptr);
  assert(called == 1);
  return result;
}

// CHECK: ThreadSanitizer: simulation stopped due to max depth
