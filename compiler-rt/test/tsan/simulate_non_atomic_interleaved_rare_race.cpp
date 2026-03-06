// RUN: %clangxx_tsan -O1 %s -o %t
// RUN: %env_tsan_opts=atexit_sleep_ms=0:abort_on_error=0:simulate_scheduler=random:simulate_iterations=1000 not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-prob100
// RUN: %env_tsan_opts=atexit_sleep_ms=0:abort_on_error=0:simulate_scheduler=random:simulate_schedule_probability=0:simulate_iterations=1000 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-prob0

// Standard TSAN rarely detect the race below. Simulation nails it very fast.

#include <atomic>
#include <pthread.h>

extern "C" int __tsan_simulate(void (*callback)(void *), void *arg);

std::atomic<int> d{};
int a = 0;

void *thread_func(void *arg) {
  ++d;
  ++a;
  ++d;
  return nullptr;
}

void test_callback(void *arg) {
  pthread_t t1, t2;
  pthread_create(&t1, nullptr, thread_func, nullptr);
  pthread_create(&t2, nullptr, thread_func, nullptr);
  pthread_join(t1, nullptr);
  pthread_join(t2, nullptr);
}

int main() { return __tsan_simulate(test_callback, nullptr); }

// CHECK-prob100: ThreadSanitizer: simulation starting
// CHECK-prob100: WARNING: ThreadSanitizer: data race
// CHECK-prob100: Write of size 4
// CHECK-prob100: Previous write of size 4

// CHECK-prob0: ThreadSanitizer: simulation starting
// CHECK-prob0-NOT: WARNING: ThreadSanitizer: data race
