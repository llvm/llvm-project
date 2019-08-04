// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <errno.h>

pthread_mutex_t m;

void *thr(void *p) {
  pthread_mutex_lock(&m);
  return 0;
}

int main() {
  pthread_mutexattr_t a;
  pthread_mutexattr_init(&a);
  pthread_mutexattr_setrobust(&a, PTHREAD_MUTEX_ROBUST);
  pthread_mutex_init(&m, &a);
  pthread_t th;
  pthread_create(&th, 0, thr, 0);
  sleep(1);
  if (pthread_mutex_lock(&m) != EOWNERDEAD) {
    fprintf(stderr, "not EOWNERDEAD\n");
    exit(1);
  }
  pthread_join(th, 0);
  fprintf(stderr, "DONE\n");
}

// This is a correct code, and tsan must not bark.
// CHECK-NOT: WARNING: ThreadSanitizer
// CHECK-NOT: EOWNERDEAD
// CHECK: DONE
// CHECK-NOT: WARNING: ThreadSanitizer

