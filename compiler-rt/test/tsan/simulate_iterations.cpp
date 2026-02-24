// RUN: %clangxx_tsan -O1 %s -o %t
// RUN: %env_tsan_opts=atexit_sleep_ms=0:abort_on_error=0:simulate_scheduler=random:simulate_iterations=1 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-1
// RUN: %env_tsan_opts=atexit_sleep_ms=0:abort_on_error=0:simulate_scheduler=random:simulate_iterations=10 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-10
// RUN: %env_tsan_opts=atexit_sleep_ms=0:abort_on_error=0:simulate_scheduler=random:simulate_iterations=100 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-100

#include <assert.h>
#include <pthread.h>
#include <stdio.h>

extern "C" int __tsan_simulate(void (*callback)(void *), void *arg);

pthread_mutex_t mutex;
int counter = 0;
int total_runs = 0;

void *thread_func(void *arg) {
  for (int i = 0; i != 10; ++i) {
    pthread_mutex_lock(&mutex);
    counter++;
    pthread_mutex_unlock(&mutex);
  }
  return nullptr;
}

void test_callback(void *arg) {
  ++total_runs;
  counter = 0;
  pthread_mutex_init(&mutex, nullptr);

  pthread_t ts[10];
  for (auto &t : ts)
    pthread_create(&t, nullptr, thread_func, nullptr);

  for (auto &t : ts)
    pthread_join(t, nullptr);

  assert(counter == 100);

  pthread_mutex_destroy(&mutex);
}

int main() {
  int result = __tsan_simulate(test_callback, nullptr);
  fprintf(stderr, "total_runs=%d\n", total_runs);
  return result;
}

// CHECK-1: ThreadSanitizer: simulation starting (iterations 0..0
// CHECK-1: total_runs=1{{$}}

// CHECK-10: ThreadSanitizer: simulation starting (iterations 0..9
// CHECK-10: total_runs=10{{$}}
//
// CHECK-100: ThreadSanitizer: simulation starting (iterations 0..9
// CHECK-100: total_runs=100{{$}}
