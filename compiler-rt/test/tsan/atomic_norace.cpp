// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
#include "test.h"

const int kTestCount = 4;
typedef long long T;
T atomics[kTestCount * 2];

void Test(int test, T *p, bool main_thread) {
  volatile T sink;
  if (test == 0) {
    if (main_thread)
      __atomic_fetch_add(p, 1, __ATOMIC_RELAXED);
    else
      __atomic_fetch_add(p, 1, __ATOMIC_RELAXED);
  } else if (test == 1) {
    if (main_thread)
      __atomic_exchange_n(p, 1, __ATOMIC_ACQ_REL);
    else
      __atomic_exchange_n(p, 1, __ATOMIC_ACQ_REL);
  } else if (test == 2) {
    if (main_thread)
      sink = __atomic_load_n(p, __ATOMIC_SEQ_CST);
    else
      __atomic_store_n(p, 1, __ATOMIC_SEQ_CST);
  } else if (test == 3) {
    if (main_thread)
      sink = __atomic_load_n(p, __ATOMIC_SEQ_CST);
    else
      sink = *p;
  }
}

void *Thread(void *p) {
  for (int i = 0; i < kTestCount; i++) {
    Test(i, &atomics[i], false);
  }
  barrier_wait(&barrier);
  barrier_wait(&barrier);
  for (int i = 0; i < kTestCount; i++) {
    fprintf(stderr, "Test %d reverse\n", i);
    Test(i, &atomics[kTestCount + i], false);
  }
  return 0;
}

int main() {
  barrier_init(&barrier, 2);
  pthread_t t;
  pthread_create(&t, 0, Thread, 0);
  barrier_wait(&barrier);
  for (int i = 0; i < kTestCount; i++) {
    fprintf(stderr, "Test %d\n", i);
    Test(i, &atomics[i], true);
  }
  for (int i = 0; i < kTestCount; i++) {
    Test(i, &atomics[kTestCount + i], true);
  }
  barrier_wait(&barrier);
  pthread_join(t, 0);
}

// CHECK-NOT: ThreadSanitizer: data race
