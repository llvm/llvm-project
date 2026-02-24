// RUN: %clangxx_tsan -O1 %s -o %t && env TSAN_OPTIONS="simulate_scheduler=random:simulate_iterations=10" %run %t 2>&1 | FileCheck %s

#include <assert.h>
#include <pthread.h>

extern "C" int __tsan_simulate(void (*callback)(void *), void *arg);

pthread_mutex_t mtx;
int shared = 0;

void *thread_func(void *arg) {
  for (int i = 0; i < 10; i++) {
    pthread_mutex_lock(&mtx);
    shared++;
    pthread_mutex_unlock(&mtx);
  }
  return nullptr;
}

void test_callback(void *) {
  shared = 0;

  pthread_mutex_init(&mtx, nullptr);

  const int kThreads = 4;
  pthread_t threads[kThreads];

  for (int i = 0; i < kThreads; i++)
    pthread_create(&threads[i], nullptr, thread_func, nullptr);

  for (int i = 0; i < kThreads; i++)
    pthread_join(threads[i], nullptr);

  assert(shared == kThreads * 10);

  pthread_mutex_destroy(&mtx);
}

int main() { return __tsan_simulate(test_callback, nullptr); }

// CHECK-NOT: WARNING: ThreadSanitizer: data race
