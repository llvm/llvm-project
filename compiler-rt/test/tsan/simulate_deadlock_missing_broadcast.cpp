// RUN: %clangxx_tsan -O1 %s -o %t
// RUN: %env_tsan_opts=atexit_sleep_ms=0:abort_on_error=0:simulate_scheduler=random:simulate_iterations=50 not %run %t 2>&1 | FileCheck %s

// Test condition variable missing broadcast deadlock.
// Scenario: Two threads wait on condition, but only one signal is sent
// Result: One waiter is left blocked forever -> deadlock

#include <assert.h>
#include <pthread.h>
#include <unistd.h>

extern "C" int __tsan_simulate(void (*callback)(void *), void *arg);

pthread_mutex_t mutex;
pthread_cond_t cond;
int c = 0;

void *thread1_func(void *arg) {
  pthread_mutex_lock(&mutex);
  c = 1;
  pthread_cond_signal(&cond); // Only wakes ONE thread!
  pthread_mutex_unlock(&mutex);
  return nullptr;
}

void *thread2_func(void *arg) {
  pthread_mutex_lock(&mutex);
  while (c != 1)
    pthread_cond_wait(&cond, &mutex);
  pthread_mutex_unlock(&mutex);
  return nullptr;
}

void *thread3_func(void *arg) {
  pthread_mutex_lock(&mutex);
  while (c != 1)
    pthread_cond_wait(&cond, &mutex);
  pthread_mutex_unlock(&mutex);
  return nullptr;
}

void test_callback(void *arg) {
  pthread_mutex_init(&mutex, nullptr);
  pthread_cond_init(&cond, nullptr);

  c = 0;
  pthread_t t1, t2, t3;
  pthread_create(&t3, nullptr, thread3_func, nullptr);
  pthread_create(&t2, nullptr, thread2_func, nullptr);
  pthread_create(&t1, nullptr, thread1_func, nullptr);

  pthread_join(t1, nullptr);
  pthread_join(t2, nullptr);
  pthread_join(t3, nullptr);

  pthread_cond_destroy(&cond);
  pthread_mutex_destroy(&mutex);
}

int main() {
  alarm(10); // Test timeout
  __tsan_simulate(test_callback, nullptr);
  assert(false);
  return 1;
}

// CHECK: ThreadSanitizer: simulation starting
// CHECK: ThreadSanitizer: deadlock detected at iteration {{[0-9]+}} - all threads are blocked
// CHECK: ThreadSanitizer: to reproduce, set TSAN_OPTIONS=simulate_scheduler=random:simulate_start_iteration={{[0-9]+}}
