// RUN: %clangxx_tsan %s -o %t
// RUN: %env_tsan_opts=atexit_sleep_ms=0:abort_on_error=0:simulate_scheduler=random:simulate_iterations=2 not %run %t 2>&1 | FileCheck %s

// Mutexes not managed by ThreadSanitizer's runtime (e.g. absl::Mutex) cannot
// be simulated correctly and may lead to deadlock during simulation.

#include <pthread.h>

extern "C" int __tsan_simulate(void (*callback)(void *arg), void *arg);
extern "C" void __tsan_mutex_pre_lock(void *m, unsigned flagz);
extern "C" void __tsan_mutex_post_lock(void *m, unsigned flagz, int rec);
extern "C" int __tsan_mutex_pre_unlock(void *m, unsigned flagz);
extern "C" void __tsan_mutex_post_unlock(void *m, unsigned flagz);

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

// CHECK: ThreadSanitizer: simulation error - unsupported interceptor called: __tsan_mutex_pre_lock
// CHECK: ThreadSanitizer: simulation error - unsupported interceptor called: __tsan_mutex_post_lock
// CHECK: ThreadSanitizer: simulation error - unsupported interceptor called: __tsan_mutex_pre_unlock
// CHECK: ThreadSanitizer: simulation error - unsupported interceptor called: __tsan_mutex_post_unlock
// CHECK: ThreadSanitizer: simulation aborted after 1 iterations
