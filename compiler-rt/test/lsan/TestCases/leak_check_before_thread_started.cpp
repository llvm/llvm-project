// Regression test for http://llvm.org/bugs/show_bug.cgi?id=21621
// This test relies on timing between threads, so any failures will be flaky.
// RUN: %clangxx_lsan %s -o %t
// RUN: %env_lsan_opts="log_pointers=1:log_threads=1" %run %t

// Fixme: remove once test passes with hwasan
// UNSUPPORTED: hwasan

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
bool flag = false;

void *func(void *arg) {
  // This mutex will never be grabbed.
  fprintf(stderr, "entered func()\n");
  pthread_mutex_lock(&mutex);
  free(arg);
  pthread_mutex_unlock(&mutex);
  return 0;
}

void create_detached_thread() {
  pthread_t thread_id;
  pthread_attr_t attr;

  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);

  void *arg = malloc(1337);
  assert(arg);
  // This mutex is never unlocked by the main thread.
  pthread_mutex_lock(&mutex);
  int res = pthread_create(&thread_id, &attr, func, arg);
  assert(res == 0);
  pthread_attr_destroy(&attr);
}

int main() {
  create_detached_thread();
}
