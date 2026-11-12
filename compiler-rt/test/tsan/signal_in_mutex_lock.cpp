// RUN: %clang_tsan %s -lstdc++ -o %t && %run %t 2>&1 | FileCheck %s

#include "test.h"
#include <pthread.h>
#include <signal.h>
#include <stdio.h>

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <mutex>

std::mutex sampler_mutex; //dummy mutex to lock in the thread we spawn.
std::mutex done_mutex;    // guards the cv and done variables.
std::condition_variable cv;
bool done = false;
std::atomic<bool> spin = true;

void *ThreadFunc(void *x) {
  while (spin) {
    // Lock the mutex
    std::lock_guard<std::mutex> guard(sampler_mutex);
    // Mutex is released at the end
  }

  return nullptr;
}

static void SigprofHandler(int signal, siginfo_t *info, void *context) {
  // Assuming we did some work, change the variable to let the main thread
  // know that we are done.
  {
    std::unique_lock<std::mutex> lck(done_mutex);
    done = true;
    cv.notify_one();
  }
}

int main() {
  alarm(60); // Kill the test if it hangs.

  // Install the signal handler
  struct sigaction sa;
  sa.sa_sigaction = SigprofHandler;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = SA_RESTART | SA_SIGINFO;
  if (sigaction(SIGPROF, &sa, 0) != 0) {
    fprintf(stderr, "failed to install signal handler\n");
    abort();
  }

  // Spawn a thread that will just loop and get the mutex lock:
  pthread_t thread;
  pthread_create(&thread, NULL, ThreadFunc, NULL);

  {
    // Lock the mutex before sending the signal
    std::lock_guard<std::mutex> guard(sampler_mutex);
    // From now on thread 1 will be waiting for the lock

    // Send the SIGPROF signal to thread.
    int r = pthread_kill(thread, SIGPROF);
    assert(r == 0);

    // Wait until signal handler sends the data.
    std::unique_lock lk(done_mutex);
    cv.wait(lk, [] { return done; });

    // We got the done variable from the signal handler. Exiting successfully.
    fprintf(stderr, "PASS\n");
  }

  // Wait for thread to prevent it from spinning on a released mutex.
  spin = false;
  pthread_join(thread, nullptr);
}

// CHECK-NOT: WARNING: ThreadSanitizer:
// CHECK: PASS
