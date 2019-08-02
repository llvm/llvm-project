// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
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
      *p = 42;
  } else if (test == 1) {
    if (main_thread)
      __atomic_fetch_add(p, 1, __ATOMIC_RELAXED);
    else
      sink = *p;
  } else if (test == 2) {
    if (main_thread)
      sink = __atomic_load_n(p, __ATOMIC_SEQ_CST);
    else
      *p = 42;
  } else if (test == 3) {
    if (main_thread)
      __atomic_store_n(p, 1, __ATOMIC_SEQ_CST);
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

// CHECK: Test 0
// CHECK: ThreadSanitizer: data race
// CHECK-NOT: SUMMARY{{.*}}tsan_interface_atomic
// CHECK: Test 1
// CHECK: ThreadSanitizer: data race
// CHECK-NOT: SUMMARY{{.*}}tsan_interface_atomic
// CHECK: Test 2
// CHECK: ThreadSanitizer: data race
// CHECK-NOT: SUMMARY{{.*}}tsan_interface_atomic
// CHECK: Test 3
// CHECK: ThreadSanitizer: data race
// CHECK-NOT: SUMMARY{{.*}}tsan_interface_atomic
// CHECK: Test 0 reverse
// CHECK: ThreadSanitizer: data race
// CHECK: Test 1 reverse
// CHECK: ThreadSanitizer: data race
// CHECK: Test 2 reverse
// CHECK: ThreadSanitizer: data race
// CHECK: Test 3 reverse
// CHECK: ThreadSanitizer: data race
