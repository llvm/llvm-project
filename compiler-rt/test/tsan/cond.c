// RUN: %clang_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
// CHECK-NOT: WARNING: ThreadSanitizer: data race
// CHECK-NOT: ThreadSanitizer WARNING: double lock
// CHECK-NOT: ThreadSanitizer WARNING: mutex unlock by another thread
// CHECK: OK

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

pthread_mutex_t m;
pthread_cond_t c;
int x;

void *thr1(void *p) {
  int i;

  for (i = 0; i < 10; i += 2) {
    pthread_mutex_lock(&m);
    while (x != i)
      pthread_cond_wait(&c, &m);
    x = i + 1;
    pthread_cond_signal(&c);
    pthread_mutex_unlock(&m);
  }
  return 0;
}

void *thr2(void *p) {
  int i;

  for (i = 1; i < 10; i += 2) {
    pthread_mutex_lock(&m);
    while (x != i)
      pthread_cond_wait(&c, &m);
    x = i + 1;
    pthread_mutex_unlock(&m);
    pthread_cond_broadcast(&c);
  }
  return 0;
}

int main() {
  pthread_t th1, th2;

  pthread_mutex_init(&m, 0);
  pthread_cond_init(&c, 0);
  pthread_create(&th1, 0, thr1, 0);
  pthread_create(&th2, 0, thr2, 0);
  pthread_join(th1, 0);
  pthread_join(th2, 0);
  fprintf(stderr, "OK\n");
}
