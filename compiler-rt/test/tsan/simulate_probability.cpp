// RUN: %clangxx_tsan -O1 %s -o %t
// RUN: %env_tsan_opts=atexit_sleep_ms=0:abort_on_error=0:simulate_scheduler=random:simulate_iterations=10:simulate_probability=0.5 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-PROB50
// RUN: %env_tsan_opts=atexit_sleep_ms=0:abort_on_error=0:simulate_scheduler=random:simulate_iterations=10:simulate_probability=1.0 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-PROB100
//
// This is a basic functional test that the parameter works; no
// validation of the probabilities are done by the test.

#include <assert.h>
#include <pthread.h>

extern "C" int __tsan_simulate(void (*callback)(void *), void *arg);

pthread_mutex_t mutex;
int counter = 0;

void *thread_func(void *arg) {
  for (int i = 0; i < 10; i++) {
    pthread_mutex_lock(&mutex);
    counter++;
    pthread_mutex_unlock(&mutex);
  }
  return nullptr;
}

void test_callback(void *arg) {
  counter = 0;
  pthread_mutex_init(&mutex, nullptr);

  pthread_t t1, t2;
  pthread_create(&t1, nullptr, thread_func, nullptr);
  pthread_create(&t2, nullptr, thread_func, nullptr);

  pthread_join(t1, nullptr);
  pthread_join(t2, nullptr);

  pthread_mutex_destroy(&mutex);

  assert(counter == 20);
}

int main() { return __tsan_simulate(test_callback, nullptr); }

// CHECK-PROB50: ThreadSanitizer: simulation starting
// CHECK-PROB50: ThreadSanitizer: simulation finished

// CHECK-PROB100: ThreadSanitizer: simulation starting
// CHECK-PROB100: ThreadSanitizer: simulation finished
