// Test that Thread objects are reused.
// RUN: %clangxx_hwasan -mllvm -hwasan-instrument-stack=0 %s -o %t && %env_hwasan_opts=verbose_threads=1 %run %t 2>&1 | FileCheck %s

#include <assert.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <sanitizer/hwasan_interface.h>

#include "utils.h"

pthread_barrier_t bar;

void *threadfn(void *) {
  pthread_barrier_wait(UNTAG(&bar));
  return nullptr;
}

void start_stop_threads() {
  constexpr int N = 2;
  pthread_t threads[N];

  pthread_barrier_init(UNTAG(&bar), nullptr, N + 1);
  for (auto &t : threads)
    pthread_create(&t, nullptr, threadfn, nullptr);

  pthread_barrier_wait(UNTAG(&bar));

  for (auto &t : threads)
    pthread_join(t, nullptr);
  pthread_barrier_destroy(UNTAG(&bar));
}

int main() {
  // Cut off initial threads.
  // CHECK: === test start ===
  fprintf(stderr, "=== test start ===\n");

  // CHECK: Creating  : T{{[0-9]+}} [[A:0x[0-9a-f]+]] stack:
  // CHECK: Creating  : T{{[0-9]+}} [[B:0x[0-9a-f]+]] stack:
  start_stop_threads();

  // CHECK-DAG: Creating  : T{{[0-9]+}} [[A]] stack:
  // CHECK-DAG: Creating  : T{{[0-9]+}} [[B]] stack:
  start_stop_threads();

  // CHECK-DAG: Creating  : T{{[0-9]+}} [[A]] stack:
  // CHECK-DAG: Creating  : T{{[0-9]+}} [[B]] stack:
  start_stop_threads();

  return 0;
}
