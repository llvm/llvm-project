// RUN: %clangxx_tsan -O1 %s -o %t
// RUN: %env_tsan_opts=atexit_sleep_ms=0:abort_on_error=0:simulate_scheduler=random:simulate_iterations=10 %run %t 2>&1 | FileCheck %s

#include <assert.h>
#include <pthread.h>

extern "C" int __tsan_simulate(void (*callback)(void *), void *arg);

const int num_mutexes = 10;
const int num_threads = 5;
pthread_mutex_t mutexes[num_mutexes];
int counter = 0;

void *thread_func(void *arg) {
  // Lock all mutexes in order
  for (int i = 0; i < num_mutexes; i++)
    pthread_mutex_lock(&mutexes[i]);

  // Critical section: increment counter
  counter++;

  // Unlock all mutexes in reverse order
  for (int i = num_mutexes - 1; i >= 0; i--)
    pthread_mutex_unlock(&mutexes[i]);

  return nullptr;
}

void test_callback(void *arg) {
  for (int i = 0; i < num_mutexes; i++)
    pthread_mutex_init(&mutexes[i], nullptr);
  counter = 0;

  pthread_t threads[num_threads];

  for (int i = 0; i < num_threads; i++)
    pthread_create(&threads[i], nullptr, thread_func, nullptr);

  for (int i = 0; i < num_threads; i++)
    pthread_join(threads[i], nullptr);

  assert(counter == num_threads);

  for (int i = 0; i < num_mutexes; i++)
    pthread_mutex_destroy(&mutexes[i]);
}

int main() { return __tsan_simulate(test_callback, nullptr); }

// CHECK: ThreadSanitizer: simulation starting
// CHECK: ThreadSanitizer: simulation finished
