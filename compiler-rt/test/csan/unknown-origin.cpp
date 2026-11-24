// RUN: %clangxx_csan -O1 -g %s -o %t -pthread
// RUN: env CSAN_OPTIONS=skip_watch=0:udelay=1000 %run %t 2>&1 | FileCheck %s

#include <atomic>
#include <pthread.h>

volatile int Global;
std::atomic<bool> Start;
std::atomic<bool> Stop;

__attribute__((no_sanitize("concurrency"))) static void *Thread(void *) {
  while (!Start.load(std::memory_order_acquire)) {
  }
  while (!Stop.load(std::memory_order_relaxed))
    ++Global;
  return nullptr;
}

int main() {
  pthread_t T;
  pthread_create(&T, nullptr, Thread, nullptr);
  Start.store(true, std::memory_order_release);
  for (int I = 0; I < 64; ++I)
    (void)Global;
  Stop.store(true, std::memory_order_relaxed);
  pthread_join(T, nullptr);
  return 0;
}

// CHECK: WARNING: ConcurrencySanitizer: data race of unknown origin
// CHECK: race at unknown origin, with read of size 4
