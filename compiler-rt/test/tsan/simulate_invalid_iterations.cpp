// RUN: %clangxx_tsan %s -o %t
// RUN: %env_tsan_opts=simulate_scheduler=random:simulate_iterations=0 not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-ZERO
// RUN: %env_tsan_opts=simulate_scheduler=random:simulate_iterations=-1 not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-NEGATIVE

#include <assert.h>
#include <pthread.h>

extern "C" int __tsan_simulate(void (*callback)(void *arg), void *arg);

void test_callback(void *arg) { assert(0); }

int main() { return __tsan_simulate(test_callback, nullptr); }

// CHECK-ZERO: ThreadSanitizer: simulate_iterations must be > 0 (got 0)

// CHECK-NEGATIVE: ThreadSanitizer: simulate_iterations must be > 0 (got -1)
