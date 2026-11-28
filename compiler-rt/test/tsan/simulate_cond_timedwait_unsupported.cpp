// RUN: %clangxx_tsan %s -o %t
// RUN: %env_tsan_opts=atexit_sleep_ms=0:abort_on_error=0:simulate_scheduler=random:simulate_iterations=2 not %run %t 2>&1 | FileCheck %s

#include <pthread.h>
#include <stdio.h>
#include <time.h>

#include <sanitizer/tsan_interface.h>

static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
static bool ready;

static void *Signaler(void *) {
  pthread_mutex_lock(&mutex);
  ready = true;
  pthread_cond_signal(&cond);
  pthread_mutex_unlock(&mutex);
  return nullptr;
}

static void Callback(void *) {
  pthread_mutex_lock(&mutex);
  ready = false;
  pthread_t t;
  pthread_create(&t, nullptr, Signaler, nullptr);
  while (!ready) {
    timespec deadline;
    clock_gettime(CLOCK_REALTIME, &deadline);
    deadline.tv_sec += 10;
    pthread_cond_timedwait(&cond, &mutex, &deadline);
  }
  pthread_mutex_unlock(&mutex);
  pthread_join(t, nullptr);
}

int main() {
  int result = __tsan_simulate(Callback, nullptr);
  fprintf(stderr, "result=%d\n", result);
  return result;
}

// CHECK: ThreadSanitizer: simulation error - unsupported interceptor called: pthread_cond_timedwait at iteration 0
// CHECK: Simulation does not support this synchronization primitive.
// CHECK: ThreadSanitizer: to reproduce, set TSAN_OPTIONS=simulate_scheduler=random:simulate_start_iteration=0
// CHECK-NOT: result=
