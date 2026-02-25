// RUN: %clangxx_tsan -O1 %s -o %t
// RUN: %env_tsan_opts=atexit_sleep_ms=0:abort_on_error=0:simulate_scheduler=random:simulate_iterations=1000 not %run %t 2>&1 | FileCheck %s

// Test based on rare_ref.cpp from https://github.com/NVIDIA/stdexec/pull/1395
// A race condition involving reference counting where two threads both access
// a non-atomic variable after decrementing the reference count.
// Standard TSAN rarely detects this race; simulation finds it quickly.

#include <atomic>
#include <pthread.h>

extern "C" int __tsan_simulate(void (*callback)(void *), void *arg);

struct TestData {
  pthread_mutex_t mtx;
  pthread_cond_t cv;
  int x = 0;
  std::atomic<int> ref{2};
  std::atomic<int> *value = new std::atomic<int>{0};
  int non_atomic = 0; // Race target
};

static void *thread1_func(void *arg) {
  TestData *data = (TestData *)arg;

  pthread_mutex_lock(&data->mtx);
  data->x = 1;
  pthread_cond_signal(&data->cv);
  pthread_mutex_unlock(&data->mtx);

  int new_ref_count = data->ref.fetch_sub(1) - 1;
  if (new_ref_count == 0) {
    delete data->value;
  }

  data->non_atomic += 1; // Race here
  return nullptr;
}

static void *thread2_func(void *arg) {
  TestData *data = (TestData *)arg;

  pthread_mutex_lock(&data->mtx);
  while (data->x == 0) {
    pthread_cond_wait(&data->cv, &data->mtx);
  }
  pthread_mutex_unlock(&data->mtx);

  int new_ref_count = data->ref.fetch_sub(1) - 1;
  if (new_ref_count == 1) {
    data->non_atomic += 1; // Race here
  }

  return nullptr;
}

void test_callback(void *arg) {
  TestData data;
  pthread_mutex_init(&data.mtx, nullptr);
  pthread_cond_init(&data.cv, nullptr);

  pthread_t t1, t2;
  pthread_create(&t1, nullptr, thread1_func, &data);
  pthread_create(&t2, nullptr, thread2_func, &data);
  pthread_join(t1, nullptr);
  pthread_join(t2, nullptr);

  delete data.value;
  pthread_mutex_destroy(&data.mtx);
  pthread_cond_destroy(&data.cv);
}

int main() { return __tsan_simulate(test_callback, nullptr); }

// CHECK: ThreadSanitizer: simulation starting
// CHECK: WARNING: ThreadSanitizer: data race
// CHECK: Write of size 4
// CHECK: Previous write of size 4
