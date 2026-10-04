// Test that TLS allocations (such as glibc's internal dlerror buffer from a
// failed dlsym call, freed in __libc_thread_freeres) on a dying detached
// thread are not falsely reported as leaks if leak checking runs after the
// sanitizer's TSD destructor has executed.
//
// RUN: %clang_lsan %s -ldl -pthread -o %t && %run %t

#define _GNU_SOURCE
#include <assert.h>
#include <dlfcn.h>
#include <limits.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <unistd.h>

static pthread_key_t key;
static atomic_int ready = 0;

static void key_destructor(void *arg) {
  uintptr_t iter = (uintptr_t)arg;
  if (iter > 1) {
    int res = pthread_setspecific(key, (void *)(iter - 1));
    assert(res == 0);
    return;
  }
  // On the last iteration (PTHREAD_DESTRUCTOR_ITERATIONS), the sanitizer's TSD
  // destructor (registered earlier with a lower key index) has already run,
  // while glibc's __libc_thread_freeres() has not yet freed the dlerror buffer.
  atomic_store(&ready, 1);
  sleep(10);
}

static void *thread_func(void *arg) {
  dlsym(RTLD_DEFAULT, "nonexistent_symbol_12345");
  int res = pthread_setspecific(key, (void *)PTHREAD_DESTRUCTOR_ITERATIONS);
  assert(res == 0);
  return NULL;
}

int main(void) {
  int res = pthread_key_create(&key, key_destructor);
  assert(res == 0);

  pthread_attr_t attr;
  res = pthread_attr_init(&attr);
  assert(res == 0);
  res = pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
  assert(res == 0);

  pthread_t th;
  res = pthread_create(&th, &attr, thread_func, NULL);
  assert(res == 0);
  pthread_attr_destroy(&attr);

  while (!atomic_load(&ready))
    sched_yield();

  return 0;
}
