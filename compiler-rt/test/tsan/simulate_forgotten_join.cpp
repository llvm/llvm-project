// RUN: %clangxx_tsan -O1 %s -o %t
// RUN: %env_tsan_opts=atexit_sleep_ms=0:abort_on_error=0:simulate_scheduler=random:simulate_iterations=9 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-NOLEAK
// RUN: %env_tsan_opts=atexit_sleep_ms=0:abort_on_error=0:simulate_scheduler=random:simulate_iterations=10 not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-LEAK

// Verify simulation detects a thread that is never joined. On the 10th
// iteration, the callback "forgets" to join one thread.

#include <pthread.h>
#include <stdio.h>

#include <sanitizer/tsan_interface.h>

static int global_count = 0;

void *thread_func(void *arg) { return nullptr; }

void test_callback(void *arg) {
  ++global_count;

  pthread_t t1, t2;
  pthread_create(&t1, nullptr, thread_func, nullptr);
  pthread_create(&t2, nullptr, thread_func, (void *)2);

  pthread_join(t1, nullptr);
  if (global_count != 10)
    pthread_join(t2, nullptr);
  // On iteration 10, t2 is never joined.
}

int main() {
  int res = __tsan_simulate(test_callback, nullptr);
  fprintf(stderr, "simulation result: %d\n", res);
  return res;
}

// CHECK-NOLEAK: ThreadSanitizer: simulation starting
// CHECK-NOLEAK: ThreadSanitizer: simulation finished (9 iterations)
// CHECK-NOLEAK: simulation result: 0

// CHECK-LEAK: ThreadSanitizer: thread leak detected at iteration 9
// CHECK-LEAK: ThreadSanitizer: simulation stopped due to thread leak
// CHECK-LEAK: simulation result: -1
