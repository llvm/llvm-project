// RUN: %clangxx_tsan -O1 %s -o %t
// RUN: %env_tsan_opts=atexit_sleep_ms=0:abort_on_error=0:simulate_scheduler=random:simulate_start_iteration=5:simulate_iterations=1 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-ITER5
// RUN: %env_tsan_opts=atexit_sleep_ms=0:abort_on_error=0:simulate_scheduler=random:simulate_start_iteration=42:simulate_iterations=3 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-ITER42

#include <pthread.h>
#include <stdio.h>

extern "C" int __tsan_simulate(void (*callback)(void *), void *arg);

pthread_mutex_t mutex;
int counter = 0;

void *thread_func(void *arg) {
  pthread_mutex_lock(&mutex);
  counter++;
  pthread_mutex_unlock(&mutex);
  return nullptr;
}

void test_callback(void *arg) {
  fprintf(stderr, "test_callback running\n");
  counter = 0;
  pthread_mutex_init(&mutex, nullptr);

  pthread_t t;
  pthread_create(&t, nullptr, thread_func, nullptr);
  pthread_join(t, nullptr);

  pthread_mutex_destroy(&mutex);
}

int main() { return __tsan_simulate(test_callback, nullptr); }

// CHECK-ITER5: ThreadSanitizer: simulation starting (iterations 5..5
// CHECK-ITER5: test_callback running
// CHECK-ITER5-NOT: test_callback running

// CHECK-ITER42: ThreadSanitizer: simulation starting (iterations 42..44
// CHECK-ITER42: test_callback running
// CHECK-ITER42: test_callback running
// CHECK-ITER42: test_callback running
// CHECK-ITER42-NOT: test_callback running
