// RUN: %clang_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
// CHECK-NOT: WARNING
// CHECK: OK
// This test is failing on powerpc64 (VMA=44). After calling pthread_cancel,
// the Thread-specific data destructors are not called, so the destructor 
// "thread_finalize" (defined in tsan_interceptors.cc) can not set the status
// of the thread to "ThreadStatusFinished" failing a check in "SetJoined" 
// (defined in sanitizer_thread_registry.cc). It might seem a bug on glibc,
// however the same version GLIBC-2.17 will not make fail the test on 
// powerpc64 BE (VMA=46)
// XFAIL: powerpc64-unknown-linux-gnu

#include "test.h"

pthread_mutex_t m;
pthread_cond_t c;
int x;

static void my_cleanup(void *arg) {
  printf("my_cleanup\n");
  pthread_mutex_unlock((pthread_mutex_t*)arg);
}

void *thr1(void *p) {
  pthread_mutex_lock(&m);
  pthread_cleanup_push(my_cleanup, &m);
  barrier_wait(&barrier);
  while (x == 0)
    pthread_cond_wait(&c, &m);
  pthread_cleanup_pop(1);
  return 0;
}

int main() {
  barrier_init(&barrier, 2);

  pthread_t th;

  pthread_mutex_init(&m, 0);
  pthread_cond_init(&c, 0);

  pthread_create(&th, 0, thr1, 0);
  barrier_wait(&barrier);
  sleep(1);  // let it block on cond var
  pthread_cancel(th);

  pthread_join(th, 0);
  pthread_mutex_lock(&m);
  pthread_mutex_unlock(&m);
  fprintf(stderr, "OK\n");
}
