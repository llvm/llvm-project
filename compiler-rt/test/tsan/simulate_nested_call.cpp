// RUN: %clangxx_tsan %s -o %t
// RUN: %env_tsan_opts=atexit_sleep_ms=0:simulate_scheduler=random:simulate_iterations=5 %run %t 2>&1 | FileCheck %s

// Verify that __tsan_simulate called from inside a simulation callback returns
// -1 without running its callback, and that the outer simulation continues.

#include <pthread.h>
#include <stdio.h>

#include <sanitizer/tsan_interface.h>

static int outer_calls;
static int inner_calls;
static int nested_failures;

static void *thread_func(void *arg) { return nullptr; }

static void inner_callback(void *arg) { inner_calls++; }

static void outer_callback(void *arg) {
  outer_calls++;
  if (__tsan_simulate(inner_callback, nullptr) == -1)
    nested_failures++;

  pthread_t t;
  pthread_create(&t, nullptr, thread_func, nullptr);
  pthread_join(t, nullptr);
}

int main() {
  int result = __tsan_simulate(outer_callback, nullptr);
  fprintf(stderr,
          "result=%d outer_calls=%d inner_calls=%d nested_failures=%d\n",
          result, outer_calls, inner_calls, nested_failures);
  return result;
}

// CHECK: ThreadSanitizer: __tsan_simulate cannot be called while a simulation is running. Not running callback
// CHECK: ThreadSanitizer: simulation finished (5 iterations)
// CHECK: result=0 outer_calls=5 inner_calls=0 nested_failures=5
