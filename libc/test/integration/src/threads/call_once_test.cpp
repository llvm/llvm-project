//===-- Tests for call_once -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/atomic.h"
#include "src/threads/call_once.h"
#include "src/threads/mtx_destroy.h"
#include "src/threads/mtx_init.h"
#include "src/threads/mtx_lock.h"
#include "src/threads/mtx_unlock.h"
#include "src/threads/thrd_create.h"
#include "src/threads/thrd_join.h"

#include "test/IntegrationTest/test.h"

#include <threads.h>

static constexpr unsigned int NUM_THREADS = 5;
static __llvm_libc::cpp::Atomic<unsigned int> thread_count;

static unsigned int call_count;
static void call_once_func() { ++call_count; }

static int func(void *) {
  static once_flag flag = ONCE_FLAG_INIT;
  __llvm_libc::call_once(&flag, call_once_func);

  thread_count.fetch_add(1);

  return 0;
}

void call_from_5_threads() {
  // Ensure the call count and thread count are 0 to begin with.
  call_count = 0;
  thread_count = 0;

  thrd_t threads[NUM_THREADS];
  for (unsigned int i = 0; i < NUM_THREADS; ++i) {
    ASSERT_EQ(__llvm_libc::thrd_create(threads + i, func, nullptr),
              static_cast<int>(thrd_success));
  }

  for (unsigned int i = 0; i < NUM_THREADS; ++i) {
    int retval;
    ASSERT_EQ(__llvm_libc::thrd_join(threads[i], &retval),
              static_cast<int>(thrd_success));
    ASSERT_EQ(retval, 0);
  }

  EXPECT_EQ(thread_count.val, 5U);
  EXPECT_EQ(call_count, 1U);
}

static mtx_t once_func_blocker;
static void blocking_once_func() {
  __llvm_libc::mtx_lock(&once_func_blocker);
  __llvm_libc::mtx_unlock(&once_func_blocker);
}

static __llvm_libc::cpp::Atomic<unsigned int> start_count;
static __llvm_libc::cpp::Atomic<unsigned int> done_count;
static int once_func_caller(void *) {
  static once_flag flag;
  start_count.fetch_add(1);
  __llvm_libc::call_once(&flag, blocking_once_func);
  done_count.fetch_add(1);
  return 0;
}

// Test the synchronization aspect of the call_once function.
// This is not a fool proof test, but something which might be
// useful when we add a flakiness detection scheme to UnitTest.
void test_synchronization() {
  start_count = 0;
  done_count = 0;

  ASSERT_EQ(__llvm_libc::mtx_init(&once_func_blocker, mtx_plain),
            static_cast<int>(thrd_success));
  // Lock the blocking mutex so that the once func blocks.
  ASSERT_EQ(__llvm_libc::mtx_lock(&once_func_blocker),
            static_cast<int>(thrd_success));

  thrd_t t1, t2;
  ASSERT_EQ(__llvm_libc::thrd_create(&t1, once_func_caller, nullptr),
            static_cast<int>(thrd_success));
  ASSERT_EQ(__llvm_libc::thrd_create(&t2, once_func_caller, nullptr),
            static_cast<int>(thrd_success));

  while (start_count.load() != 2)
    ; // Spin until both threads start.

  // Since the once func is blocked, the threads should not be done yet.
  EXPECT_EQ(done_count.val, 0U);

  // Unlock the blocking mutex so that the once func blocks.
  ASSERT_EQ(__llvm_libc::mtx_unlock(&once_func_blocker),
            static_cast<int>(thrd_success));

  int retval;
  ASSERT_EQ(__llvm_libc::thrd_join(t1, &retval),
            static_cast<int>(thrd_success));
  ASSERT_EQ(retval, 0);
  ASSERT_EQ(__llvm_libc::thrd_join(t2, &retval),
            static_cast<int>(thrd_success));
  ASSERT_EQ(retval, 0);

  ASSERT_EQ(done_count.val, 2U);

  __llvm_libc::mtx_destroy(&once_func_blocker);
}

TEST_MAIN() {
  call_from_5_threads();
  test_synchronization();
  return 0;
}
