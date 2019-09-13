// Test that lsan handles tls correctly for many threads
// RUN: LSAN_BASE="report_objects=1:use_stacks=0:use_registers=0"
// RUN: %clangxx_lsan %s -DUSE_THREAD -o %t
// RUN: %env_lsan_opts=$LSAN_BASE:"use_tls=0" not %run %t 2>&1 | FileCheck %s
// RUN: %env_lsan_opts=$LSAN_BASE:"use_tls=1" %run %t 2>&1
// RUN: %env_lsan_opts="" %run %t 2>&1

// Patch r303906 did not fix all the problems.
// UNSUPPORTED: arm-linux,armhf-linux

#include <assert.h>
#include <limits.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>

static const int NUM_THREADS = 10;

pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
int finished = 0;

__thread void *ptr1;
__thread void *ptr2;
__thread void *ptr3;
__thread void *ptr4;
__thread void *ptr5;

void alloc() {
  ptr1 = malloc(1111);
  ptr2 = malloc(2222);
  ptr3 = malloc(3333);
  ptr4 = malloc(4444);
  ptr5 = malloc(5555);
}

void *thread_start(void *arg) {
  alloc();

  pthread_mutex_lock(&mutex);
  finished++;
  pthread_mutex_unlock(&mutex);

  // don't exit, to intentionally leak tls data
  while (1)
    sleep(100);
}

int main() {
  pthread_t thread[NUM_THREADS];
  for (int i = 0; i < NUM_THREADS; ++i) {
    assert(0 == pthread_create(&thread[i], 0, thread_start, 0));
  }
  // spin until all threads have finished
  while (finished < NUM_THREADS)
    sleep(1);
  exit(0);
}

// CHECK: LeakSanitizer: detected memory leaks
// CHECK: SUMMARY: {{(Leak|Address)}}Sanitizer:
