// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
#include "test.h"

// OS X doesn't have pthread_setname_np(tid, name).
// UNSUPPORTED: darwin

#if defined(__FreeBSD__)
#include <pthread_np.h>
#define pthread_setname_np pthread_set_name_np
#endif

long long Global;

void *Thread1(void *x) {
  barrier_wait(&barrier);
  Global++;
  return 0;
}

void *Thread2(void *x) {
  pthread_setname_np(pthread_self(), "foobar2");
  Global--;
  barrier_wait(&barrier);
  return 0;
}

int main() {
  barrier_init(&barrier, 3);
  pthread_t t[2];
  pthread_create(&t[0], 0, Thread1, 0);
  pthread_create(&t[1], 0, Thread2, 0);
  pthread_setname_np(t[0], "foobar1");
  barrier_wait(&barrier);
  pthread_join(t[0], NULL);
  pthread_join(t[1], NULL);
}

// CHECK: WARNING: ThreadSanitizer: data race
// CHECK:   Thread T1 'foobar1'
// CHECK:   Thread T2 'foobar2'
