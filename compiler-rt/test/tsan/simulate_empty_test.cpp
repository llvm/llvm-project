// RUN: %clangxx_tsan -O1 %s -o %t
// RUN: %env_tsan_opts=atexit_sleep_ms=0:simulate_scheduler=random %run %t 2>&1 | FileCheck %s

#include <assert.h>
#include <pthread.h>
#include <stdio.h>

extern "C" int __tsan_simulate(void (*callback)(void *), void *arg);

static int called;
void test_callback(void *arg) {
  ++called;
  fprintf(stderr, "Callback executed with no threads\n");
}

int main() {
  int result = __tsan_simulate(test_callback, nullptr);
  assert(called == 1);
  return result;
}

// CHECK: ThreadSanitizer: simulation starting (iterations 0..
// CHECK: Callback executed with no threads
// CHECK: ThreadSanitizer: simulation exiting - no threads were spawned
