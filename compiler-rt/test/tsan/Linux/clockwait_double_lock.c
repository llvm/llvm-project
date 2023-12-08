// Regression test for https://github.com/google/sanitizers/issues/1259
// RUN: %clang_tsan -O1 %s -o %t && %run %t
// REQUIRES: glibc-2.30 || android-30

#define _GNU_SOURCE
#include <pthread.h>

pthread_cond_t cv;
pthread_mutex_t mtx;

void *fn(void *vp) {
  pthread_mutex_lock(&mtx);
  pthread_cond_signal(&cv);
  pthread_mutex_unlock(&mtx);
  return NULL;
}

int main() {
  pthread_mutex_lock(&mtx);

  pthread_t tid;
  pthread_create(&tid, NULL, fn, NULL);

  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  ts.tv_sec += 10;
  pthread_cond_clockwait(&cv, &mtx, CLOCK_MONOTONIC, &ts);
  pthread_mutex_unlock(&mtx);

  pthread_join(tid, NULL);
  return 0;
}
