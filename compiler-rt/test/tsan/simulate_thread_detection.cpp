// RUN: %clangxx_tsan -O1 %s -o %t
// RUN: %env_tsan_opts=simulate_scheduler=random:simulate_iterations=2 %run %t 2>&1 | FileCheck %s

#include "test.h"
#include <assert.h>
#include <atomic>
#include <stdio.h>

extern "C" int __tsan_simulate(void (*callback)(void *arg), void *arg);

std::atomic<bool> keep_running(true);

void *background_thread(void *arg) {
  while (keep_running.load(std::memory_order_relaxed)) {
    usleep(10000);
  }
  return nullptr;
}

void test_callback(void *arg) {
  long test_case = (long)arg;
  fprintf(stderr, "test_callback test_case=%ld\n", test_case);
}

int main() {
  pthread_t bg;
  pthread_create(&bg, nullptr, background_thread, nullptr);

  assert(__tsan_simulate(test_callback, (void *)1) != 0);

  keep_running.store(false, std::memory_order_relaxed);
  pthread_join(bg, nullptr);

  assert(__tsan_simulate(test_callback, (void *)2) == 0);
  return 0;
}

// CHECK: ThreadSanitizer: simulation cannot start - other threads are running
// CHECK: Simulation requires that only the calling thread exists
// CHECK-NOT: test_callback test_case=1
// CHECK: ThreadSanitizer: simulation starting (iterations 0..1
// CHECK: test_callback test_case=2
// CHECK: ThreadSanitizer: simulation exiting - no threads were spawned
