//===-- Unittests for call_once -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/threads.h"
#include "src/threads/call_once.h"
#include "src/threads/mtx_init.h"
#include "src/threads/mtx_lock.h"
#include "src/threads/mtx_unlock.h"
#include "src/threads/thrd_create.h"
#include "src/threads/thrd_join.h"
#include "utils/UnitTest/Test.h"

#include <stdatomic.h>

static constexpr unsigned int num_threads = 5;
static atomic_uint thread_count;

static unsigned int call_count;
static void call_once_func() { ++call_count; }

static int func(void *) {
  static once_flag flag = ONCE_FLAG_INIT;
  __llvm_libc::call_once(&flag, call_once_func);

  ++thread_count; // This is a an atomic update.

  return 0;
}

TEST(CallOnceTest, CallFrom5Threads) {
  // Ensure the call count and thread count are 0 to begin with.
  call_count = 0;
  thread_count = 0;

  thrd_t threads[num_threads];
  for (unsigned int i = 0; i < num_threads; ++i) {
    ASSERT_EQ(__llvm_libc::thrd_create(threads + i, func, nullptr),
              static_cast<int>(thrd_success));
  }

  for (unsigned int i = 0; i < num_threads; ++i) {
    int retval;
    ASSERT_EQ(__llvm_libc::thrd_join(threads + i, &retval),
              static_cast<int>(thrd_success));
    ASSERT_EQ(retval, 0);
  }

  EXPECT_EQ(static_cast<unsigned int>(thread_count), 5U);
  EXPECT_EQ(call_count, 1U);
}

static mtx_t once_func_blocker;
static void blocking_once_func() {
  __llvm_libc::mtx_lock(&once_func_blocker);
  __llvm_libc::mtx_unlock(&once_func_blocker);
}

static atomic_uint start_count;
static atomic_uint done_count;
static int once_func_caller(void *) {
  static once_flag flag;
  ++start_count;
  __llvm_libc::call_once(&flag, blocking_once_func);
  ++done_count;
  return 0;
}

// Test the synchronization aspect of the call_once function.
// This is not a fool proof test, but something which might be
// useful when we add a flakiness detection scheme to UnitTest.
TEST(CallOnceTest, TestSynchronization) {
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

  while (start_count != 2)
    ; // Spin until both threads start.

  // Since the once func is blocked, the threads should not be done yet.
  EXPECT_EQ(static_cast<unsigned int>(done_count), 0U);

  // Unlock the blocking mutex so that the once func blocks.
  ASSERT_EQ(__llvm_libc::mtx_unlock(&once_func_blocker),
            static_cast<int>(thrd_success));

  int retval;
  ASSERT_EQ(__llvm_libc::thrd_join(&t1, &retval),
            static_cast<int>(thrd_success));
  ASSERT_EQ(retval, 0);
  ASSERT_EQ(__llvm_libc::thrd_join(&t2, &retval),
            static_cast<int>(thrd_success));
  ASSERT_EQ(retval, 0);

  ASSERT_EQ(static_cast<unsigned int>(done_count), 2U);
}
