// RUN: %clangxx_tsan -O1 %s -o %t
// RUN: %env_tsan_opts=atexit_sleep_ms=0:abort_on_error=0:simulate_scheduler=random:simulate_iterations=20 %run %t 2>&1 | FileCheck %s

#include <assert.h>
#include <pthread.h>

extern "C" int __tsan_simulate(void (*callback)(void *), void *arg);

pthread_mutex_t mutex;
int counter = 0;

void *thread_func(void *arg) {
  for (int i = 0; i < 50; i++) {
    pthread_mutex_lock(&mutex);
    counter++;
    pthread_mutex_unlock(&mutex);
  }
  return nullptr;
}

void test_callback(void *arg) {
  counter = 0;
  pthread_mutex_init(&mutex, nullptr);

  const int num_threads = 8;
  pthread_t threads[num_threads];

  for (int i = 0; i < num_threads; i++) {
    pthread_create(&threads[i], nullptr, thread_func, nullptr);
  }

  for (int i = 0; i < num_threads; i++) {
    pthread_join(threads[i], nullptr);
  }

  pthread_mutex_destroy(&mutex);

  assert(counter == num_threads * 50);
}

int main() { return __tsan_simulate(test_callback, nullptr); }

// CHECK: ThreadSanitizer: simulation starting
// CHECK: ThreadSanitizer: simulation finished
