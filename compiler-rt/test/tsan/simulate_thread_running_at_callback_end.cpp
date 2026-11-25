// RUN: %clangxx_tsan -O1 %s -o %t
// RUN: %env_tsan_opts=atexit_sleep_ms=0:simulate_scheduler=random:simulate_iterations=20:simulate_schedule_probability=0 %run %t detach 2>&1 | FileCheck %s --check-prefix=DETACHED
// RUN: %env_tsan_opts=atexit_sleep_ms=0:simulate_scheduler=random:simulate_iterations=20 %run %t detach 2>&1 | FileCheck %s --check-prefix=DETACHED
// RUN: %env_tsan_opts=atexit_sleep_ms=0:abort_on_error=0:simulate_scheduler=random:simulate_iterations=5:simulate_schedule_probability=0 not %run %t 2>&1 | FileCheck %s --check-prefix=LEAK

// Verify that a thread which has not finished when the callback returns runs
// to completion before the next iteration starts or __tsan_simulate returns.
// With simulate_schedule_probability=0 the callback never yields, so the
// thread it creates has not started when the callback returns.
//
// "detach": each iteration creates a detached thread. Every thread must
// finish within its own iteration.
// Default: each iteration creates a joinable thread and never joins it. The
// thread must finish and be reported as leaked at iteration 0.

#include <atomic>
#include <pthread.h>
#include <stdio.h>
#include <string.h>

#include <sanitizer/tsan_interface.h>

static bool detach_threads;
static std::atomic<int> started_iterations;
static std::atomic<int> finished_threads;
static int overlapping_iterations;

static void *Worker(void *) {
  finished_threads.fetch_add(1);
  return nullptr;
}

static void Callback(void *) {
  int iteration = started_iterations.fetch_add(1);
  if (finished_threads.load() != iteration)
    overlapping_iterations++;

  pthread_t thread;
  pthread_create(&thread, nullptr, Worker, nullptr);
  if (detach_threads)
    pthread_detach(thread);
}

int main(int argc, char **argv) {
  detach_threads = argc > 1 && strcmp(argv[1], "detach") == 0;
  int result = __tsan_simulate(Callback, nullptr);
  fprintf(stderr, "result=%d threads_finished=%d overlapping_iterations=%d\n",
          result, finished_threads.load(), overlapping_iterations);
  return result;
}

// DETACHED: ThreadSanitizer: simulation finished (20 iterations)
// DETACHED: result=0 threads_finished=20 overlapping_iterations=0

// LEAK: ThreadSanitizer: thread leak detected at iteration 0
// LEAK: ThreadSanitizer: simulation stopped due to thread leak after 1 iterations
// LEAK: result=-1 threads_finished=1 overlapping_iterations=0
