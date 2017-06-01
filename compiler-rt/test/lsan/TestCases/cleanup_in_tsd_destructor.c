// Regression test for thread lifetime tracking. Thread data should be
// considered live during the thread's termination, at least until the
// user-installed TSD destructors have finished running (since they may contain
// additional cleanup tasks). LSan doesn't actually meet that goal 100%, but it
// makes its best effort.
// RUN: LSAN_BASE="report_objects=1:use_registers=0:use_stacks=0:use_globals=0"
// RUN: %clang_lsan %s -o %t
// RUN: LSAN_OPTIONS=$LSAN_BASE:use_tls=1 %run %t
// RUN: LSAN_OPTIONS=$LSAN_BASE:use_tls=0 not %run %t 2>&1 | FileCheck %s

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#include "sanitizer/lsan_interface.h"
#include "sanitizer_common/print_address.h"

pthread_key_t key;
__thread void *p;

void key_destructor(void *arg) {
  // Generally this may happen on a different thread.
  __lsan_do_leak_check();
}

void *thread_func(void *arg) {
  p = malloc(1337);
  print_address("Test alloc: ", 1, p);
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
// CHECK: Test alloc: [[ADDR:0x[0-9,a-f]+]]
// CHECK: [[ADDR]] (1337 bytes)
