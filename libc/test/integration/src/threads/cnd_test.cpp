//===-- Tests for standard condition variables ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/atomic.h"
#include "src/threads/cnd_broadcast.h"
#include "src/threads/cnd_destroy.h"
#include "src/threads/cnd_init.h"
#include "src/threads/cnd_signal.h"
#include "src/threads/cnd_wait.h"
#include "src/threads/mtx_destroy.h"
#include "src/threads/mtx_init.h"
#include "src/threads/mtx_lock.h"
#include "src/threads/mtx_unlock.h"
#include "src/threads/thrd_create.h"
#include "src/threads/thrd_join.h"

#include "test/IntegrationTest/test.h"

#include <threads.h>

namespace wait_notify_broadcast_test {

// The test in this namespace tests all condition variable operations. The
// main thread spawns THRD_COUNT threads, each of which wait on a condition
// variable |broadcast_cnd|. After spawing the threads, it waits on another
// condition variable |threads_ready_cnd| which will be signalled by the last
// thread before it starts waiting on |broadcast_cnd|. On signalled by the
// last thread, the main thread then wakes up to broadcast to all waiting
// threads to wake up. Each of the THRD_COUNT child threads increment
// |broadcast_count| by 1 before they start waiting on |broadcast_cnd|, and
// decrement it by 1 after getting signalled on |broadcast_cnd|.

constexpr unsigned int THRD_COUNT = 1000;

static LIBC_NAMESPACE::cpp::Atomic<unsigned int> broadcast_count(0);
static cnd_t broadcast_cnd, threads_ready_cnd;
static mtx_t broadcast_mtx, threads_ready_mtx;

int broadcast_thread_func(void *) {
  LIBC_NAMESPACE::mtx_lock(&broadcast_mtx);
  unsigned oldval = broadcast_count.fetch_add(1);
  if (oldval == THRD_COUNT - 1) {
    LIBC_NAMESPACE::mtx_lock(&threads_ready_mtx);
    LIBC_NAMESPACE::cnd_signal(&threads_ready_cnd);
    LIBC_NAMESPACE::mtx_unlock(&threads_ready_mtx);
  }

  LIBC_NAMESPACE::cnd_wait(&broadcast_cnd, &broadcast_mtx);
  LIBC_NAMESPACE::mtx_unlock(&broadcast_mtx);
  broadcast_count.fetch_sub(1);
  return 0;
}

void wait_notify_broadcast_test() {
  LIBC_NAMESPACE::cnd_init(&broadcast_cnd);
  LIBC_NAMESPACE::cnd_init(&threads_ready_cnd);
  LIBC_NAMESPACE::mtx_init(&broadcast_mtx, mtx_plain);
  LIBC_NAMESPACE::mtx_init(&threads_ready_mtx, mtx_plain);

  LIBC_NAMESPACE::mtx_lock(&threads_ready_mtx);
  thrd_t threads[THRD_COUNT];
  for (unsigned int i = 0; i < THRD_COUNT; ++i)
    LIBC_NAMESPACE::thrd_create(&threads[i], broadcast_thread_func, nullptr);

  LIBC_NAMESPACE::cnd_wait(&threads_ready_cnd, &threads_ready_mtx);
  LIBC_NAMESPACE::mtx_unlock(&threads_ready_mtx);

  LIBC_NAMESPACE::mtx_lock(&broadcast_mtx);
  ASSERT_EQ(broadcast_count.val, THRD_COUNT);
  LIBC_NAMESPACE::cnd_broadcast(&broadcast_cnd);
  LIBC_NAMESPACE::mtx_unlock(&broadcast_mtx);

  for (unsigned int i = 0; i < THRD_COUNT; ++i) {
    int retval = 0xBAD;
    LIBC_NAMESPACE::thrd_join(threads[i], &retval);
    ASSERT_EQ(retval, 0);
  }

  ASSERT_EQ(broadcast_count.val, 0U);

  LIBC_NAMESPACE::cnd_destroy(&broadcast_cnd);
  LIBC_NAMESPACE::cnd_destroy(&threads_ready_cnd);
  LIBC_NAMESPACE::mtx_destroy(&broadcast_mtx);
  LIBC_NAMESPACE::mtx_destroy(&threads_ready_mtx);
}

} // namespace wait_notify_broadcast_test

namespace single_waiter_test {

// In this namespace we set up test with two threads, one the main thread
// and the other a waiter thread. They wait on each other using condition
// variables and mutexes before proceeding to completion.

mtx_t waiter_mtx, main_thread_mtx;
cnd_t waiter_cnd, main_thread_cnd;

int waiter_thread_func([[maybe_unused]] void *unused) {
  LIBC_NAMESPACE::mtx_lock(&waiter_mtx);

  LIBC_NAMESPACE::mtx_lock(&main_thread_mtx);
  LIBC_NAMESPACE::cnd_signal(&main_thread_cnd);
  LIBC_NAMESPACE::mtx_unlock(&main_thread_mtx);

  LIBC_NAMESPACE::cnd_wait(&waiter_cnd, &waiter_mtx);
  LIBC_NAMESPACE::mtx_unlock(&waiter_mtx);

  return 0x600D;
}

void single_waiter_test() {
  ASSERT_EQ(LIBC_NAMESPACE::mtx_init(&waiter_mtx, mtx_plain),
            int(thrd_success));
  ASSERT_EQ(LIBC_NAMESPACE::mtx_init(&main_thread_mtx, mtx_plain),
            int(thrd_success));
  ASSERT_EQ(LIBC_NAMESPACE::cnd_init(&waiter_cnd), int(thrd_success));
  ASSERT_EQ(LIBC_NAMESPACE::cnd_init(&main_thread_cnd), int(thrd_success));

  ASSERT_EQ(LIBC_NAMESPACE::mtx_lock(&main_thread_mtx), int(thrd_success));

  thrd_t waiter_thread;
  LIBC_NAMESPACE::thrd_create(&waiter_thread, waiter_thread_func, nullptr);

  ASSERT_EQ(LIBC_NAMESPACE::cnd_wait(&main_thread_cnd, &main_thread_mtx),
            int(thrd_success));
  ASSERT_EQ(LIBC_NAMESPACE::mtx_unlock(&main_thread_mtx), int(thrd_success));

  ASSERT_EQ(LIBC_NAMESPACE::mtx_lock(&waiter_mtx), int(thrd_success));
  ASSERT_EQ(LIBC_NAMESPACE::cnd_signal(&waiter_cnd), int(thrd_success));
  ASSERT_EQ(LIBC_NAMESPACE::mtx_unlock(&waiter_mtx), int(thrd_success));

  int retval;
  LIBC_NAMESPACE::thrd_join(waiter_thread, &retval);
  ASSERT_EQ(retval, 0x600D);

  LIBC_NAMESPACE::mtx_destroy(&waiter_mtx);
  LIBC_NAMESPACE::mtx_destroy(&main_thread_mtx);
  LIBC_NAMESPACE::cnd_destroy(&waiter_cnd);
  LIBC_NAMESPACE::cnd_destroy(&main_thread_cnd);
}

} // namespace single_waiter_test

TEST_MAIN() {
  wait_notify_broadcast_test::wait_notify_broadcast_test();
  single_waiter_test::single_waiter_test();
  return 0;
}
