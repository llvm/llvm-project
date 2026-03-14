// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <errno.h>

pthread_mutex_t m;
int x;

void *thr(void *p) {
  pthread_mutex_lock(&m);
  x = 42;
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
  if (pthread_mutex_trylock(&m) != EOWNERDEAD) {
    fprintf(stderr, "not EOWNERDEAD\n");
    exit(1);
  }
  x = 43;
  pthread_join(th, 0);
  fprintf(stderr, "DONE\n");
}

// This is a false positive, tsan must not bark at the data race.
// But currently it does.
// CHECK-NOT: WARNING: ThreadSanitizer WARNING: double lock of mutex
// CHECK: WARNING: ThreadSanitizer: data race
// CHECK-NOT: EOWNERDEAD
// CHECK: DONE
// CHECK-NOT: WARNING: ThreadSanitizer

