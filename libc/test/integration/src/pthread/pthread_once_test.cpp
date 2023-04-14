//===-- Tests for pthread_once --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/atomic.h"
#include "src/pthread/pthread_create.h"
#include "src/pthread/pthread_join.h"
#include "src/pthread/pthread_mutex_destroy.h"
#include "src/pthread/pthread_mutex_init.h"
#include "src/pthread/pthread_mutex_lock.h"
#include "src/pthread/pthread_mutex_unlock.h"
#include "src/pthread/pthread_once.h"

#include "test/IntegrationTest/test.h"

#include <pthread.h>
#include <stdint.h> // uintptr_t

static constexpr unsigned int NUM_THREADS = 5;
static __llvm_libc::cpp::Atomic<unsigned int> thread_count;

static unsigned int call_count;
static void pthread_once_func() { ++call_count; }

static void *func(void *) {
  static pthread_once_t flag = PTHREAD_ONCE_INIT;
  ASSERT_EQ(__llvm_libc::pthread_once(&flag, pthread_once_func), 0);

  thread_count.fetch_add(1);

  return nullptr;
}

void call_from_5_threads() {
  // Ensure the call count and thread count are 0 to begin with.
  call_count = 0;
  thread_count = 0;

  pthread_t threads[NUM_THREADS];
  for (unsigned int i = 0; i < NUM_THREADS; ++i) {
    ASSERT_EQ(__llvm_libc::pthread_create(threads + i, nullptr, func, nullptr),
              0);
  }

  for (unsigned int i = 0; i < NUM_THREADS; ++i) {
    void *retval;
    ASSERT_EQ(__llvm_libc::pthread_join(threads[i], &retval), 0);
    ASSERT_EQ(uintptr_t(retval), uintptr_t(0));
  }

  EXPECT_EQ(thread_count.val, 5U);
  EXPECT_EQ(call_count, 1U);
}

static pthread_mutex_t once_func_blocker;
static void blocking_once_func() {
  __llvm_libc::pthread_mutex_lock(&once_func_blocker);
  __llvm_libc::pthread_mutex_unlock(&once_func_blocker);
}

static __llvm_libc::cpp::Atomic<unsigned int> start_count;
static __llvm_libc::cpp::Atomic<unsigned int> done_count;
static void *once_func_caller(void *) {
  static pthread_once_t flag;
  start_count.fetch_add(1);
  __llvm_libc::pthread_once(&flag, blocking_once_func);
  done_count.fetch_add(1);
  return nullptr;
}

// Test the synchronization aspect of the pthread_once function.
// This is not a fool proof test, but something which might be
// useful when we add a flakiness detection scheme to UnitTest.
void test_synchronization() {
  start_count = 0;
  done_count = 0;

  ASSERT_EQ(__llvm_libc::pthread_mutex_init(&once_func_blocker, nullptr), 0);
  // Lock the blocking mutex so that the once func blocks.
  ASSERT_EQ(__llvm_libc::pthread_mutex_lock(&once_func_blocker), 0);

  pthread_t t1, t2;
  ASSERT_EQ(
      __llvm_libc::pthread_create(&t1, nullptr, once_func_caller, nullptr), 0);
  ASSERT_EQ(
      __llvm_libc::pthread_create(&t2, nullptr, once_func_caller, nullptr), 0);

  while (start_count.load() != 2)
    ; // Spin until both threads start.

  // Since the once func is blocked, the threads should not be done yet.
  EXPECT_EQ(done_count.val, 0U);

  // Unlock the blocking mutex so that the once func blocks.
  ASSERT_EQ(__llvm_libc::pthread_mutex_unlock(&once_func_blocker), 0);

  void *retval;
  ASSERT_EQ(__llvm_libc::pthread_join(t1, &retval), uintptr_t(0));
  ASSERT_EQ(uintptr_t(retval), 0);
  ASSERT_EQ(__llvm_libc::pthread_join(t2, &retval), uintptr_t(0));
  ASSERT_EQ(uintptr_t(retval), 0);

  ASSERT_EQ(done_count.val, 2U);

  __llvm_libc::pthread_mutex_destroy(&once_func_blocker);
}

TEST_MAIN() {
  call_from_5_threads();
  test_synchronization();
  return 0;
}
