// RUN: %clangxx_tsan -O1 %s -o %t && env TSAN_OPTIONS="simulate_scheduler=random:simulate_iterations=10" %run %t 2>&1 | FileCheck %s

#include <pthread.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

#include <sanitizer/tsan_interface.h>

void *thread_func(void *arg) {
  sleep(1);
  usleep(1000);
  struct timespec ts = {0, 1000000};
  nanosleep(&ts, nullptr);
  return nullptr;
}

void test_callback(void *arg) {
  pthread_t t;
  pthread_create(&t, nullptr, thread_func, nullptr);
  pthread_join(t, nullptr);
}

int main() {
  int res = __tsan_simulate(test_callback, nullptr);
  fprintf(stderr, "simulation result: %d\n", res);
  return res;
}

// CHECK: ThreadSanitizer: simulation starting
// CHECK: simulation result: 0
