// RUN: %clang_tsan -O1 %s -lpthread -o %t && %deflake %run %t | FileCheck %s
// Regression test for
// https://github.com/google/sanitizers/issues/468
// When the data race was reported, pthread_atfork() handler used to be
// executed which caused another race report in the same thread, which resulted
// in a deadlock.
#include "test.h"

int glob = 0;

void *worker(void *unused) {
  barrier_wait(&barrier);
  glob++;
  return NULL;
}

void atfork() {
  fprintf(stderr, "ATFORK\n");
  glob++;
}

int main() {
  barrier_init(&barrier, 2);
  pthread_atfork(atfork, atfork, atfork);
  pthread_t t;
  pthread_create(&t, NULL, worker, NULL);
  glob++;
  barrier_wait(&barrier);
  pthread_join(t, NULL);
  // CHECK: ThreadSanitizer: data race
  // CHECK-NOT: ATFORK
  return 0;
}
