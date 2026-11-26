// RUN: %clangxx_tsan %s -o %t
// RUN: %env_tsan_opts=atexit_sleep_ms=0:abort_on_error=0:simulate_scheduler=random:simulate_iterations=2 not %run %t 2>&1 | FileCheck %s

// Mutexes not managed by ThreadSanitizer's runtime (e.g. absl::Mutex) cannot
// be simulated correctly and may lead to deadlock during simulation.

#include <pthread.h>

#include <sanitizer/tsan_interface.h>

int fake_mutex;

void *thread_func(void *arg) {
  __tsan_mutex_pre_lock(&fake_mutex, 0);
  __tsan_mutex_post_lock(&fake_mutex, 0, 0);
  __tsan_mutex_pre_unlock(&fake_mutex, 0);
  __tsan_mutex_post_unlock(&fake_mutex, 0);
  return nullptr;
}

void test_callback(void *arg) {
  pthread_t t;
  pthread_create(&t, nullptr, thread_func, nullptr);
  pthread_join(t, nullptr);
}

int main() { return __tsan_simulate(test_callback, nullptr); }

// CHECK: ThreadSanitizer: simulation error - unsupported interceptor called: __tsan_mutex_pre_lock at iteration 0
// CHECK: ThreadSanitizer: to reproduce, set TSAN_OPTIONS=simulate_scheduler=random:simulate_start_iteration=0
// CHECK-NOT: __tsan_mutex_post_lock
