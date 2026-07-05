// Regression test. Disabler should not depend on TSD validity.
// RUN: %clang_lsan %s -o %t
// RUN: %env_lsan_opts="report_objects=1:use_registers=0:use_stacks=0:use_tls=1:use_ld_allocations=0" %run %t

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#include "sanitizer/lsan_interface.h"

pthread_key_t key;

void key_destructor(void *arg) {
  __lsan_disable();
  void *p = malloc(1337);
  // Break optimization.
  fprintf(stderr, "Test alloc: %p.\n", p);
  pthread_setspecific(key, 0);
  __lsan_enable();
}

void *thread_func(void *arg) {
  int res = pthread_setspecific(key, (void*)1);
  assert(res == 0);
  return 0;
}

int main() {
  int res = pthread_key_create(&key, &key_destructor);
  assert(res == 0);
  pthread_t thread_id;
  res = pthread_create(&thread_id, 0, thread_func, 0);
  assert(res == 0);
  res = pthread_join(thread_id, 0);
  assert(res == 0);
  return 0;
}
