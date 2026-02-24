// RUN: %clangxx_tsan -O1 %s -o %t
// RUN: %env_tsan_opts=atexit_sleep_ms=0:abort_on_error=0:simulate_scheduler=random:simulate_iterations=5 %run %t 2>&1 | FileCheck %s
//
// Test thread_local object destruction.
// Verifies that thread_local destructors are called and can safely decrement atomics.

#include <assert.h>
#include <atomic>
#include <pthread.h>

extern "C" int __tsan_simulate(void (*callback)(void *), void *arg);

std::atomic<int> ctor_count(0);
std::atomic<int> dtor_count(0);

class ThreadLocalObject {
public:
  ThreadLocalObject() { ctor_count.fetch_add(1, std::memory_order_relaxed); }

  ~ThreadLocalObject() { dtor_count.fetch_add(1, std::memory_order_relaxed); }
};

void *thread_func(void *arg) {
  // Access thread_local variable to trigger construction
  thread_local ThreadLocalObject obj;
  return nullptr;
}

void test_callback(void *arg) {
  ctor_count.store(0, std::memory_order_relaxed);
  dtor_count.store(0, std::memory_order_relaxed);

  pthread_t t;
  pthread_create(&t, nullptr, thread_func, nullptr);
  pthread_join(t, nullptr);
  pthread_create(&t, nullptr, thread_func, nullptr);
  pthread_join(t, nullptr);
  pthread_create(&t, nullptr, thread_func, nullptr);
  pthread_join(t, nullptr);

  assert(ctor_count.load(std::memory_order_relaxed) == 3);
  assert(dtor_count.load(std::memory_order_relaxed) == 3);
}

int main() { return __tsan_simulate(test_callback, nullptr); }

// CHECK: ThreadSanitizer: simulation starting
