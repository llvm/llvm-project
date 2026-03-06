// RUN: %clangxx_tsan %s -o %t
// RUN: %env_tsan_opts=simulate_scheduler=random:simulate_iterations=10:simulate_start_iteration=-1 not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-NEG1
// RUN: %env_tsan_opts=simulate_scheduler=random:simulate_iterations=10:simulate_start_iteration=-5 not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-NEG5

#include <assert.h>
#include <pthread.h>

extern "C" int __tsan_simulate(void (*callback)(void *arg), void *arg);

void test_callback(void *arg) { assert(0); }

int main() { return __tsan_simulate(test_callback, nullptr); }

// CHECK-NEG1: ThreadSanitizer: simulate_start_iteration must be >= 0 (got -1)

// CHECK-NEG5: ThreadSanitizer: simulate_start_iteration must be >= 0 (got -5)
