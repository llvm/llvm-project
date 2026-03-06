// RUN: %clangxx_tsan %s -o %t
// RUN: %env_tsan_opts=atexit_sleep_ms=0:abort_on_error=0:simulate_scheduler=random:simulate_iterations=2 not %run %t 2>&1 | FileCheck %s
//
// pthread_mutex_timedlock is not available on Apple
// UNSUPPORTED: darwin

#include <pthread.h>
#include <stdlib.h>
#include <time.h>

extern "C" int __tsan_simulate(void (*callback)(void *arg), void *arg);

pthread_mutex_t mutex;

void *thread_func(void *arg) {
  // This should trigger the unsupported interceptor error
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  ts.tv_sec += 1; // 1 second timeout

  pthread_mutex_timedlock(&mutex, &ts);
  pthread_mutex_unlock(&mutex);
  return nullptr;
}

void test_callback(void *arg) {
  pthread_mutex_init(&mutex, nullptr);

  pthread_t t;
  pthread_create(&t, nullptr, thread_func, nullptr);
  pthread_join(t, nullptr);

  pthread_mutex_destroy(&mutex);
}

int main() { return __tsan_simulate(test_callback, nullptr); }

// CHECK: ThreadSanitizer: simulation error - unsupported interceptor called: pthread_mutex_timedlock
// CHECK: Simulation does not support this synchronization primitive
// CHECK: ThreadSanitizer: simulation aborted after 1 iterations
