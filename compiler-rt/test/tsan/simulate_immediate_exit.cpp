// RUN: %clangxx_tsan -O1 %s -o %t
// RUN: %env_tsan_opts=atexit_sleep_ms=0:abort_on_error=0:simulate_scheduler=random:simulate_iterations=10 %run %t 2>&1 | FileCheck %s

#include <pthread.h>
#include <stdio.h>

extern "C" int __tsan_simulate(void (*callback)(void *), void *arg);

void *thread_func(void *arg) { return nullptr; }

void test_callback(void *arg) {
  pthread_t threads[5];

  for (int i = 0; i < 5; i++) {
    pthread_create(&threads[i], nullptr, thread_func, nullptr);
  }

  for (int i = 0; i < 5; i++) {
    pthread_join(threads[i], nullptr);
  }

  fprintf(stderr, "All immediate-exit threads joined successfully\n");
}

int main() { return __tsan_simulate(test_callback, nullptr); }

// CHECK: ThreadSanitizer: simulation starting
// CHECK: All immediate-exit threads joined successfully
// CHECK: All immediate-exit threads joined successfully
// CHECK: All immediate-exit threads joined successfully
// CHECK: All immediate-exit threads joined successfully
// CHECK: All immediate-exit threads joined successfully
// CHECK: All immediate-exit threads joined successfully
// CHECK: All immediate-exit threads joined successfully
// CHECK: All immediate-exit threads joined successfully
// CHECK: All immediate-exit threads joined successfully
// CHECK: All immediate-exit threads joined successfully
// CHECK-NOT: All immediate-exit threads joined successfully
// CHECK: ThreadSanitizer: simulation finished
