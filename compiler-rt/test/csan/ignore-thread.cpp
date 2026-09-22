// RUN: %clangxx_csan -O1 %s -o %t -pthread
// RUN: env CSAN_OPTIONS=skip_watch=0:udelay=1000 %run %t 2>&1 | FileCheck %s

#include <atomic>
#include <pthread.h>
#include <stdio.h>

extern "C" void __csan_ignore_thread_begin();
extern "C" void __csan_ignore_thread_end();
extern "C" unsigned long long __csan_get_num_data_races();

volatile int Global;
std::atomic<bool> Start;
std::atomic<bool> Stop;

__attribute__((no_sanitize("concurrency"))) static void *Thread(void *) {
  Start.store(true, std::memory_order_release);
  while (!Stop.load(std::memory_order_relaxed))
    ++Global;
  return nullptr;
}

int main() {
  pthread_t T;
  pthread_create(&T, nullptr, Thread, nullptr);
  while (!Start.load(std::memory_order_acquire)) {
  }

  __csan_ignore_thread_begin();
  __csan_ignore_thread_begin();
  for (int I = 0; I < 1024; ++I)
    (void)Global;
  __csan_ignore_thread_end();
  for (int I = 0; I < 1024; ++I)
    (void)Global;
  __csan_ignore_thread_end();

  Stop.store(true, std::memory_order_relaxed);
  pthread_join(T, nullptr);
  printf("races: %llu\n", __csan_get_num_data_races());
  return 0;
}

// CHECK-NOT: WARNING: ConcurrencySanitizer
// CHECK: races: 0
