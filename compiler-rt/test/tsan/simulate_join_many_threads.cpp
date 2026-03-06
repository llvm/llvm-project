// RUN: %clangxx_tsan -O1 %s -o %t
// RUN: %env_tsan_opts=atexit_sleep_ms=0:abort_on_error=0:simulate_scheduler=random:simulate_iterations=10 %run %t 2>&1 | FileCheck %s

// Test nested thread join chain scenario - should work correctly.
// Scenario: T1 creates/joins T2, T2 creates/joins T3, ... up to 16 levels.

#include <assert.h>
#include <pthread.h>

extern "C" int __tsan_simulate(void (*callback)(void *), void *arg);

constexpr int kMaxLevels = 16;

int counter = 0;
pthread_mutex_t mutex;

struct ThreadArg {
  int level;
};

void *thread_chain_func(void *arg) {
  ThreadArg *thread_arg = static_cast<ThreadArg *>(arg);
  if (thread_arg->level >= kMaxLevels)
    return nullptr;

  pthread_mutex_lock(&mutex);
  counter++;
  pthread_mutex_unlock(&mutex);

  pthread_t child;
  ThreadArg child_arg = {thread_arg->level + 1};
  pthread_create(&child, nullptr, thread_chain_func, &child_arg);
  pthread_join(child, nullptr);
  return nullptr;
}

void test_callback(void *arg) {
  counter = 0;
  pthread_t root;
  ThreadArg root_arg = {1};

  pthread_mutex_init(&mutex, nullptr);
  pthread_create(&root, nullptr, thread_chain_func, &root_arg);
  pthread_join(root, nullptr);
  pthread_mutex_destroy(&mutex);

  assert(counter == 15);
}

int main() { return __tsan_simulate(test_callback, nullptr); }

// CHECK: ThreadSanitizer: simulation starting
