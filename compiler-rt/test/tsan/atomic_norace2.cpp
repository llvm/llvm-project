// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
#include "test.h"

const int kTestCount = 3;
typedef long long T;
T data[kTestCount];
T atomics[kTestCount];

void *Thread(void *p) {
  for (int i = 0; i < kTestCount; i++) {
    barrier_wait(&barrier);
    while (__atomic_load_n(&atomics[i], __ATOMIC_ACQUIRE) == 0) {
    }
    data[i]++;
  }
  return 0;
}

int main() {
  barrier_init(&barrier, 2);
  pthread_t t;
  pthread_create(&t, 0, Thread, 0);
  for (int i = 0; i < kTestCount; i++) {
    barrier_wait(&barrier);
    usleep(100 * 1000);
    data[i] = 1;
    switch (i) {
    case 0:
      __atomic_store_n(&atomics[i], 1, __ATOMIC_RELEASE);
      break;
    case 1:
      __atomic_fetch_add(&atomics[1], 1, __ATOMIC_RELEASE);
      break;
    case 2:
      T cmp = 0;
      __atomic_compare_exchange_n(&atomics[2], &cmp, 1, true, __ATOMIC_RELEASE, __ATOMIC_RELAXED);
      break;
    }
  }
  pthread_join(t, 0);
  fprintf(stderr, "DONE\n");
}

// CHECK-NOT: ThreadSanitizer: data race
// CHECK: DONE
