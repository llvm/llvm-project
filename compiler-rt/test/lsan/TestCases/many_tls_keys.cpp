// Test that lsan handles tls correctly for many threads
// RUN: LSAN_BASE="report_objects=1:use_stacks=0:use_registers=0"
// RUN: %clangxx_lsan %s -DUSE_THREAD -o %t-thread
// RUN: %clangxx_lsan %s -DUSE_PTHREAD -o %t-pthread
// RUN: %env_lsan_opts=$LSAN_BASE:"use_tls=0" not %run %t-thread 2>&1 | FileCheck %s
// RUN: %env_lsan_opts=$LSAN_BASE:"use_tls=1" %run %t-thread 2>&1
// RUN: %env_lsan_opts="" %run %t-thread 2>&1
// RUN: %env_lsan_opts=$LSAN_BASE:"use_tls=0" not %run %t-pthread 2>&1 | FileCheck %s
// RUN: %env_lsan_opts=$LSAN_BASE:"use_tls=1" %run %t-pthread 2>&1
// RUN: %env_lsan_opts="" %run %t-pthread 2>&1

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

#if USE_THREAD
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

#elif USE_PTHREAD
// We won't be able to create the maximum number of keys, due to other users
// of the tls, but we'll use as many keys as we can before failing to create
// a new key.
pthread_key_t keys[PTHREAD_KEYS_MAX];
static const int PTHREAD_KEY_INVALID = 0xffffffff;

void alloc() {
  for (int i = 0; i < PTHREAD_KEYS_MAX; ++i) {
    void *ptr = malloc(123);
    if ((keys[i] == PTHREAD_KEY_INVALID) || pthread_setspecific(keys[i], ptr)) {
      free(ptr);
      break;
    }
  }
}

void pthread_destructor(void *arg) {
  assert(0 && "pthread destructors shouldn't be called");
}
#endif

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
#if USE_PTHREAD
  for (int i = 0; i < PTHREAD_KEYS_MAX; ++i) {
    if (pthread_key_create(&keys[i], pthread_destructor)) {
      keys[i] = PTHREAD_KEY_INVALID;
      break;
    }
  }
#endif

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
