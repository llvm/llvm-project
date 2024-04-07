// RUN: %clangxx_tsan -O1 %s %link_libcxx_tsan -o %t && %run %t 2>&1
// This is a correct program and tsan should not report a race.
//
// Verify that there is a happens-before relationship between a
// memory_order_release store that happens as part of a successful
// compare_exchange_strong(), and an atomic_thread_fence(memory_order_acquire)
// that happens after a relaxed load.

#include <atomic>
#include <sanitizer/tsan_interface.h>
#include <stdbool.h>
#include <stdio.h>
#include <thread>

std::atomic<bool> a;
unsigned int b;
constexpr int loops = 100000;

void Thread1() {
  for (int i = 0; i < loops; ++i) {
    while (a.load(std::memory_order_acquire)) {
    }
    b = i;
    bool expected = false;
    a.compare_exchange_strong(expected, true, std::memory_order_acq_rel);
  }
}

int main() {
  std::thread t(Thread1);
  unsigned int sum = 0;
  for (int i = 0; i < loops; ++i) {
    while (!a.load(std::memory_order_relaxed)) {
    }
    std::atomic_thread_fence(std::memory_order_acquire);
    __tsan_acquire(&a);
    sum += b;
    a.store(false, std::memory_order_release);
  }
  t.join();
  fprintf(stderr, "DONE: %u\n", sum);
  return 0;
}
