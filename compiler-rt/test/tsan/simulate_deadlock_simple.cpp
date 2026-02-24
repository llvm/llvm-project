// RUN: %clangxx_tsan -O1 %s -o %t
// RUN: %env_tsan_opts=atexit_sleep_ms=0:abort_on_error=0:simulate_scheduler=random:simulate_iterations=50 not %run %t 2>&1 | FileCheck %s
//
// Test simple deadlock potential detection: 2 threads, 2 mutexes, circular dependency.
// Thread 1: lock(A) -> lock(B)
// Thread 2: lock(B) -> lock(A)
// TSAN should detect the lock-order-inversion (potential deadlock).

#include <pthread.h>
#include <unistd.h>

extern "C" int __tsan_simulate(void (*callback)(void *), void *arg);

pthread_mutex_t mutex_a;
pthread_mutex_t mutex_b;

void *thread1_func(void *arg) {
  pthread_mutex_lock(&mutex_a);
  pthread_mutex_lock(&mutex_b);

  pthread_mutex_unlock(&mutex_b);
  pthread_mutex_unlock(&mutex_a);
  return nullptr;
}

void *thread2_func(void *arg) {
  pthread_mutex_lock(&mutex_b);
  pthread_mutex_lock(&mutex_a);

  pthread_mutex_unlock(&mutex_a);
  pthread_mutex_unlock(&mutex_b);
  return nullptr;
}

void test_callback(void *arg) {
  pthread_mutex_init(&mutex_a, nullptr);
  pthread_mutex_init(&mutex_b, nullptr);

  pthread_t t1, t2;
  pthread_create(&t1, nullptr, thread1_func, nullptr);
  pthread_create(&t2, nullptr, thread2_func, nullptr);

  pthread_join(t1, nullptr);
  pthread_join(t2, nullptr);

  pthread_mutex_destroy(&mutex_a);
  pthread_mutex_destroy(&mutex_b);
}

int main() { return __tsan_simulate(test_callback, nullptr); }

// CHECK: ThreadSanitizer: simulation starting
// CHECK: WARNING: ThreadSanitizer: lock-order-inversion (potential deadlock)
// CHECK: Cycle in lock order graph
