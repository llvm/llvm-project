// RUN: %clangxx_tsan -O1 %s -o %t
// RUN: %env_tsan_opts=atexit_sleep_ms=0:abort_on_error=0:simulate_scheduler=random:simulate_iterations=10 %run %t 2>&1 | FileCheck %s
//
// Test threads creating other threads (nested thread creation).
// Verifies that thread tracking handles hierarchical thread creation.

#include <assert.h>
#include <pthread.h>

extern "C" int __tsan_simulate(void (*callback)(void *), void *arg);

pthread_mutex_t mutex;
int counter = 0;

void *level3_func(void *arg) {
  pthread_mutex_lock(&mutex);
  counter++;
  pthread_mutex_unlock(&mutex);
  return nullptr;
}

void *level2_func(void *arg) {
  pthread_mutex_lock(&mutex);
  counter++;
  pthread_mutex_unlock(&mutex);

  // Level 2 creates Level 3
  pthread_t t;
  pthread_create(&t, nullptr, level3_func, nullptr);
  pthread_join(t, nullptr);

  return nullptr;
}

void *level1_func(void *arg) {
  pthread_mutex_lock(&mutex);
  counter++;
  pthread_mutex_unlock(&mutex);

  // Level 1 creates Level 2
  pthread_t t;
  pthread_create(&t, nullptr, level2_func, nullptr);
  pthread_join(t, nullptr);

  return nullptr;
}

void test_callback(void *arg) {
  counter = 0;
  pthread_mutex_init(&mutex, nullptr);

  // Main creates Level 1
  pthread_t t;
  pthread_create(&t, nullptr, level1_func, nullptr);
  pthread_join(t, nullptr);

  pthread_mutex_destroy(&mutex);

  assert(counter == 3);
}

int main() { return __tsan_simulate(test_callback, nullptr); }

// CHECK: ThreadSanitizer: simulation starting
// CHECK: ThreadSanitizer: simulation finished
