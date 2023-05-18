// Test that lsan handles tls correctly for many threads
// RUN: %clangxx_lsan %s -o %t
// RUN: %env_lsan_opts="report_objects=1:use_stacks=0:use_registers=0:use_tls=0" not %run %t 2>&1 | FileCheck %s
// RUN: %env_lsan_opts="report_objects=1:use_stacks=0:use_registers=0:use_tls=1" %run %t 2>&1
// RUN: %env_lsan_opts="" %run %t 2>&1

// On glibc, this requires the range returned by GetTLS to include
// specific_1stblock and specific in `struct pthread`.
// UNSUPPORTED: arm-linux, armhf-linux

// TSD on NetBSD does not use TLS
// UNSUPPORTED: target={{.*netbsd.*}}

#include <assert.h>
#include <limits.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>

static const int NUM_THREADS = 10;

pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
int finished = 0;

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
  for (int i = 0; i < PTHREAD_KEYS_MAX; ++i) {
    if (pthread_key_create(&keys[i], pthread_destructor)) {
      keys[i] = PTHREAD_KEY_INVALID;
      break;
    }
  }

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
// CHECK: SUMMARY: {{.*}}Sanitizer:
