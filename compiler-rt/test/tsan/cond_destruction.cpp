// RUN: %clangxx_tsan -O1 %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s
// RUN: %run %t arg 2>&1 | FileCheck %s
// RUN: %run %t arg arg 2>&1 | FileCheck %s
#include "test.h"

// Test for destruction of pthread_cond_t.
// POSIX states that it is safe  to destroy a condition variable upon which no
// threads are currently blocked. That is, it is not necessary to wait untill
// other threads return from pthread_cond_wait, they just need to be unblocked.

pthread_mutex_t m;
pthread_cond_t c;
bool done1, done2;

void *thr(void *p) {
  pthread_mutex_lock(&m);
  done1 = true;
  pthread_cond_signal(&c);
  while (!done2)
    pthread_cond_wait(&c, &m);
  pthread_mutex_unlock(&m);
  return 0;
}

int main(int argc, char **argv) {
  pthread_t th;
  pthread_mutex_init(&m, 0);
  pthread_cond_init(&c, 0);
  pthread_create(&th, 0, thr, 0);
  pthread_mutex_lock(&m);
  while (!done1)
    pthread_cond_wait(&c, &m);
  done2 = true;
  // Any of these sequences is legal.
  if (argc == 1) {
    pthread_cond_signal(&c);
    pthread_mutex_unlock(&m);
    pthread_cond_destroy(&c);
  } else if (argc == 2) {
    pthread_mutex_unlock(&m);
    pthread_cond_signal(&c);
    pthread_cond_destroy(&c);
  } else {
    pthread_cond_signal(&c);
    pthread_cond_destroy(&c);
    pthread_mutex_unlock(&m);
  }
  pthread_join(th, 0);
  fprintf(stderr, "DONE\n");
}

// CHECK-NOT: ThreadSanitizer: data race
