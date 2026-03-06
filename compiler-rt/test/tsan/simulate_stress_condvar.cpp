// RUN: %clangxx_tsan -O1 %s -o %t
// RUN: %env_tsan_opts=atexit_sleep_ms=0:abort_on_error=0:simulate_scheduler=random:simulate_iterations=5 %run %t 2>&1 | FileCheck %s

#include <assert.h>
#include <pthread.h>

extern "C" int __tsan_simulate(void (*callback)(void *), void *arg);

pthread_mutex_t mutex;
pthread_cond_t condvar;
int ready = 0;
int workers_done = 0;

void *worker_thread(void *arg) {
  pthread_mutex_lock(&mutex);

  while (!ready) {
    pthread_cond_wait(&condvar, &mutex);
  }

  workers_done++;
  pthread_mutex_unlock(&mutex);

  return nullptr;
}

void test_callback(void *arg) {
  ready = 0;
  workers_done = 0;
  pthread_mutex_init(&mutex, nullptr);
  pthread_cond_init(&condvar, nullptr);

  const int num_workers = 3;
  pthread_t threads[num_workers];

  for (int i = 0; i < num_workers; i++)
    pthread_create(&threads[i], nullptr, worker_thread, nullptr);

  pthread_mutex_lock(&mutex);
  ready = 1;
  pthread_cond_broadcast(&condvar);
  pthread_mutex_unlock(&mutex);

  for (int i = 0; i < num_workers; i++)
    pthread_join(threads[i], nullptr);

  pthread_cond_destroy(&condvar);
  pthread_mutex_destroy(&mutex);

  assert(workers_done == num_workers);
}

int main() { return __tsan_simulate(test_callback, nullptr); }

// CHECK: ThreadSanitizer: simulation starting
// CHECK: ThreadSanitizer: simulation finished
