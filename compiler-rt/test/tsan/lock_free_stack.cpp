// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_tsan -O1 %s -DRACE -o %t && %deflake %run %t | FileCheck %s --check-prefix=CHECK-RACE

// Found in the post-submit testing under PPC (documented in
// https://reviews.llvm.org/D110552), this test fails under PowerPC. Should be
// investigated at some point.
// UNSUPPORTED: ppc

#include "test.h"

const int kThreadCount = 4;
#if RACE
const int kTestCount = 16;
#else
const int kTestCount = 9;
#endif

template <typename F> F for_each_mo(int mo, F f) {
  f(mo);
  return f;
}

template <typename... Rest>
auto for_each_mo(int mo, Rest... rest) -> decltype(for_each_mo(rest...)) {
  auto f = for_each_mo(rest...);
  f(mo);
  return f;
}

void LockFreeStackImpl(int test, bool main_thread, int mo2, int mo4) {
  struct Node {
    int data;
    Node *next;
  };
  static Node *heads[kTestCount]{};
  auto head = heads + test;

  auto concurrent_push = [head](Node *new_head, int mo1, int mo2, int mo3) {
    auto expected = __atomic_load_n(head, mo1);
    do {
      new_head->next = expected;
    } while (!__atomic_compare_exchange_n(head, &expected, new_head,
                                          /*weak*/ true, mo2, mo3));
  };

  auto concurrent_grab_all = [head](int mo4) {
    volatile int sink{};
    (void)sink;

    auto h = __atomic_exchange_n(head, nullptr, mo4);
    while (h) {
      sink = ++h->data;
      auto next = h->next;
      delete h;
      h = next;
    }
  };

  if (main_thread) {
    concurrent_grab_all(mo4);
  } else {
    int i = 0;
    // We have 15 combinations of mo1 and mo3. Since we have two race reports
    // for each combination (the first report is for 'data' and the second
    // report for 'next'), there are 30 race reports in total that should match
    // to "CHECK-RACE-COUNT{-number_of_reports}" below
    for_each_mo(
        __ATOMIC_RELAXED, __ATOMIC_ACQUIRE, __ATOMIC_SEQ_CST, [&](int mo1) {
          for_each_mo(__ATOMIC_RELAXED, __ATOMIC_ACQUIRE, __ATOMIC_RELEASE,
                      __ATOMIC_ACQ_REL, __ATOMIC_SEQ_CST, [&](int mo3) {
                        concurrent_push(new Node{i++}, mo1, mo2, mo3);
                      });
        });
  }
}

void LockFreeStack(int test, bool main_thread, int mo2, int mo4) {
  barrier_wait(&barrier);
  if (main_thread) {
    // We need to call LockFreeStackImpl second time after the barrier
    // to guarantee at least one grab_all and cleanup.
    // However, it is better to have one instantiation of LockFreeStackImpl
    // on the main thread to merge the call stacks and prevent double race
    // reports. Therefore, we use two interation for loop and skip the barrier
    // on the second iteration.
    for (int i = 0; i < 2; ++i) {
      LockFreeStackImpl(test, main_thread, mo2, mo4);
      if (i == 0) {
        barrier_wait(&barrier);
      }
    }
  } else {
    LockFreeStackImpl(test, main_thread, mo2, mo4);
    barrier_wait(&barrier);
  }
}

void Test(bool main_thread) {
  for (int test = 0; test < kTestCount; test++) {
    if (main_thread) {
      fprintf(stderr, "Test %d\n", test);
    }
    switch (test) {
#if RACE
    case 0:
      LockFreeStack(test, main_thread, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
      break;
    case 1:
      LockFreeStack(test, main_thread, __ATOMIC_RELAXED, __ATOMIC_ACQUIRE);
      break;
    case 2:
      LockFreeStack(test, main_thread, __ATOMIC_RELAXED, __ATOMIC_RELEASE);
      break;
    case 3:
      LockFreeStack(test, main_thread, __ATOMIC_RELAXED, __ATOMIC_ACQ_REL);
      break;
    case 4:
      LockFreeStack(test, main_thread, __ATOMIC_RELAXED, __ATOMIC_SEQ_CST);
      break;
    case 5:
      LockFreeStack(test, main_thread, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED);
      break;
    case 6:
      LockFreeStack(test, main_thread, __ATOMIC_ACQUIRE, __ATOMIC_ACQUIRE);
      break;
    case 7:
      LockFreeStack(test, main_thread, __ATOMIC_ACQUIRE, __ATOMIC_RELEASE);
      break;
    case 8:
      LockFreeStack(test, main_thread, __ATOMIC_ACQUIRE, __ATOMIC_ACQ_REL);
      break;
    case 9:
      LockFreeStack(test, main_thread, __ATOMIC_ACQUIRE, __ATOMIC_SEQ_CST);
      break;
    case 10:
      LockFreeStack(test, main_thread, __ATOMIC_RELEASE, __ATOMIC_RELAXED);
      break;
    case 11:
      LockFreeStack(test, main_thread, __ATOMIC_RELEASE, __ATOMIC_RELEASE);
      break;
    case 12:
      LockFreeStack(test, main_thread, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED);
      break;
    case 13:
      LockFreeStack(test, main_thread, __ATOMIC_ACQ_REL, __ATOMIC_RELEASE);
      break;
    case 14:
      LockFreeStack(test, main_thread, __ATOMIC_SEQ_CST, __ATOMIC_RELAXED);
      break;
    case 15:
      LockFreeStack(test, main_thread, __ATOMIC_SEQ_CST, __ATOMIC_RELEASE);
      break;
#else
    case 0:
      LockFreeStack(test, main_thread, __ATOMIC_RELEASE, __ATOMIC_ACQUIRE);
      break;
    case 1:
      LockFreeStack(test, main_thread, __ATOMIC_RELEASE, __ATOMIC_ACQ_REL);
      break;
    case 2:
      LockFreeStack(test, main_thread, __ATOMIC_RELEASE, __ATOMIC_SEQ_CST);
      break;
    case 3:
      LockFreeStack(test, main_thread, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
      break;
    case 4:
      LockFreeStack(test, main_thread, __ATOMIC_ACQ_REL, __ATOMIC_ACQ_REL);
      break;
    case 5:
      LockFreeStack(test, main_thread, __ATOMIC_ACQ_REL, __ATOMIC_SEQ_CST);
      break;
    case 6:
      LockFreeStack(test, main_thread, __ATOMIC_SEQ_CST, __ATOMIC_ACQUIRE);
      break;
    case 7:
      LockFreeStack(test, main_thread, __ATOMIC_SEQ_CST, __ATOMIC_ACQ_REL);
      break;
    case 8:
      LockFreeStack(test, main_thread, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
      break;
#endif
    }
  }
}

void *Thread(void *p) {
  Test(false);
  return 0;
}

int main() {
  barrier_init(&barrier, kThreadCount);
  pthread_t t[kThreadCount - 1];
  for (int i = 0; i < kThreadCount - 1; ++i)
    pthread_create(t + i, 0, Thread, (void *)(uintptr_t)(i + 1));
  Test(true);
  for (int i = 0; i < kThreadCount - 1; ++i)
    pthread_join(t[i], 0);
}

// No race tests
// CHECK-NOT: ThreadSanitizer: data race

// Race tests
// 30 is the number of race reports for 15 possible combinations of mo1 and mo3
// CHECK-RACE: Test 0
// CHECK-RACE-COUNT-30: SUMMARY: ThreadSanitizer: data race
// CHECK-RACE-NOT: SUMMARY: ThreadSanitizer: data race
// CHECK-RACE: Test 1
// CHECK-RACE-COUNT-30: SUMMARY: ThreadSanitizer: data race
// CHECK-RACE-NOT: SUMMARY: ThreadSanitizer: data race
// CHECK-RACE: Test 2
// CHECK-RACE-COUNT-30: SUMMARY: ThreadSanitizer: data race
// CHECK-RACE-NOT: SUMMARY: ThreadSanitizer: data race
// CHECK-RACE: Test 3
// CHECK-RACE-COUNT-30: SUMMARY: ThreadSanitizer: data race
// CHECK-RACE-NOT: SUMMARY: ThreadSanitizer: data race
// CHECK-RACE: Test 4
// CHECK-RACE-COUNT-30: SUMMARY: ThreadSanitizer: data race
// CHECK-RACE-NOT: SUMMARY: ThreadSanitizer: data race
// CHECK-RACE: Test 5
// CHECK-RACE-COUNT-30: SUMMARY: ThreadSanitizer: data race
// CHECK-RACE-NOT: SUMMARY: ThreadSanitizer: data race
// CHECK-RACE: Test 6
// CHECK-RACE-COUNT-30: SUMMARY: ThreadSanitizer: data race
// CHECK-RACE-NOT: SUMMARY: ThreadSanitizer: data race
// CHECK-RACE: Test 7
// CHECK-RACE-COUNT-30: SUMMARY: ThreadSanitizer: data race
// CHECK-RACE-NOT: SUMMARY: ThreadSanitizer: data race
// CHECK-RACE: Test 8
// CHECK-RACE-COUNT-30: SUMMARY: ThreadSanitizer: data race
// CHECK-RACE-NOT: SUMMARY: ThreadSanitizer: data race
// CHECK-RACE: Test 9
// CHECK-RACE-COUNT-30: SUMMARY: ThreadSanitizer: data race
// CHECK-RACE-NOT: SUMMARY: ThreadSanitizer: data race
// CHECK-RACE: Test 10
// CHECK-RACE-COUNT-30: SUMMARY: ThreadSanitizer: data race
// CHECK-RACE-NOT: SUMMARY: ThreadSanitizer: data race
// CHECK-RACE: Test 11
// CHECK-RACE-COUNT-30: SUMMARY: ThreadSanitizer: data race
// CHECK-RACE-NOT: SUMMARY: ThreadSanitizer: data race
// CHECK-RACE: Test 12
// CHECK-RACE-COUNT-30: SUMMARY: ThreadSanitizer: data race
// CHECK-RACE-NOT: SUMMARY: ThreadSanitizer: data race
// CHECK-RACE: Test 13
// CHECK-RACE-COUNT-30: SUMMARY: ThreadSanitizer: data race
// CHECK-RACE-NOT: SUMMARY: ThreadSanitizer: data race
// CHECK-RACE: Test 14
// CHECK-RACE-COUNT-30: SUMMARY: ThreadSanitizer: data race
// CHECK-RACE-NOT: SUMMARY: ThreadSanitizer: data race
// CHECK-RACE: Test 15
// CHECK-RACE-COUNT-30: SUMMARY: ThreadSanitizer: data race
// CHECK-RACE-NOT: SUMMARY: ThreadSanitizer: data race
