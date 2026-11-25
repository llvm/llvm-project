// RUN: %clangxx_tsan %s -o %t
// RUN: %env_tsan_opts=simulate_scheduler=radnom not %run %t 2>&1 | FileCheck %s

// Verify that an unknown simulate_scheduler value is an error and the
// callback does not run.

#include <stdio.h>

#include <sanitizer/tsan_interface.h>

static int callback_calls;

void test_callback(void *arg) { callback_calls++; }

int main() {
  int result = __tsan_simulate(test_callback, nullptr);
  fprintf(stderr, "result=%d callback_calls=%d\n", result, callback_calls);
  return result;
}

// CHECK: ThreadSanitizer: unknown simulate_scheduler 'radnom' (supported values: random). Not running callback
// CHECK: result=-1 callback_calls=0
