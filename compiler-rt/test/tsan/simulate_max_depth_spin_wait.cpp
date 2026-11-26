// RUN: %clangxx_tsan -O1 %s -o %t
// RUN: %env_tsan_opts=atexit_sleep_ms=0:abort_on_error=0:simulate_scheduler=random:simulate_iterations=20:simulate_max_depth=50:simulate_schedule_probability=100 not %run %t 2>&1 | FileCheck %s --check-prefix=IN-CALLBACK
// RUN: %env_tsan_opts=atexit_sleep_ms=0:abort_on_error=0:simulate_scheduler=random:simulate_iterations=20:simulate_max_depth=50:simulate_schedule_probability=100 not %run %t after_callback 2>&1 | FileCheck %s --check-prefix=AFTER-CALLBACK

#include <atomic>
#include <pthread.h>
#include <stdio.h>
#include <string.h>

#include <sanitizer/tsan_interface.h>

static bool after_callback;
static std::atomic<int> progress;
static std::atomic<bool> done;
static std::atomic<bool> release;

static void *Worker(void *) {
  for (int i = 0; i < 200; i++)
    progress.fetch_add(1, std::memory_order_relaxed);
  done.store(true);
  return nullptr;
}

static void *SpinUntilReleased(void *) {
  while (!release.load())
    progress.fetch_add(1, std::memory_order_relaxed);
  return nullptr;
}

static void Callback(void *) {
  progress.store(0);
  done.store(false);
  release.store(false);
  pthread_t t;
  if (after_callback) {
    pthread_create(&t, nullptr, SpinUntilReleased, nullptr);
    pthread_detach(t);
    return;
  }
  pthread_create(&t, nullptr, Worker, nullptr);
  while (!done.load()) {
  }
  pthread_join(t, nullptr);
}

int main(int argc, char **argv) {
  after_callback = argc > 1 && strcmp(argv[1], "after_callback") == 0;
  int result = __tsan_simulate(Callback, nullptr);
  fprintf(stderr, "result=%d\n", result);
  return result;
}

// IN-CALLBACK: ThreadSanitizer: simulation hit max depth 50 at iteration 0
// IN-CALLBACK-NOT: Threads created by the callback
// IN-CALLBACK: ThreadSanitizer: to reproduce, set TSAN_OPTIONS=simulate_scheduler=random:simulate_start_iteration=0
// IN-CALLBACK-NOT: result=

// AFTER-CALLBACK: ThreadSanitizer: simulation hit max depth 50 at iteration 0
// AFTER-CALLBACK: Threads created by the callback were still running after the callback returned.
// AFTER-CALLBACK: ThreadSanitizer: to reproduce, set TSAN_OPTIONS=simulate_scheduler=random:simulate_start_iteration=0
// AFTER-CALLBACK-NOT: result=
