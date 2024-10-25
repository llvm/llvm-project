// RUN: %clang_lsan %s -o %t && %env_lsan_opts=log_threads=1 %run %t 2>&1 | FileCheck %s

// XFAIL: hwasan

// No pthread barriers on Darwin.
// UNSUPPORTED: darwin

#include <assert.h>
#include <pthread.h>
#include <sanitizer/lsan_interface.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

pthread_barrier_t bar;

void *threadfn(void *arg) {
  pthread_barrier_wait(&bar);
  sleep(10000);
  return 0;
}

int main(int argc, char *argv[]) {
  pthread_t thread_id;
  pthread_barrier_init(&bar, 0, 3);

  pthread_create(&thread_id, 0, threadfn, 0);
  pthread_create(&thread_id, 0, threadfn, 0);

  pthread_barrier_wait(&bar);
  return 0;
}

// CHECK: Thread T0/{{[0-9]+}} was created by T-1
// CHECK: Thread T1/{{[0-9]+}} was created by T0/
// CHECK: Thread T2/{{[0-9]+}} was created by T0/
