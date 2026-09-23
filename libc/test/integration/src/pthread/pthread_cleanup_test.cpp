//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Tests for pthread_cleanup_push and pthread_cleanup_pop.
///
//===----------------------------------------------------------------------===//

#include "src/pthread/__pthread_cleanup_pop.h"
#include "src/pthread/__pthread_cleanup_push.h"
#include "src/pthread/pthread_create.h"
#include "src/pthread/pthread_exit.h"
#include "src/pthread/pthread_getspecific.h"
#include "src/pthread/pthread_join.h"
#include "src/pthread/pthread_key_create.h"
#include "src/pthread/pthread_key_delete.h"
#include "src/pthread/pthread_mutex_destroy.h"
#include "src/pthread/pthread_mutex_init.h"
#include "src/pthread/pthread_mutex_lock.h"
#include "src/pthread/pthread_mutex_trylock.h"
#include "src/pthread/pthread_mutex_unlock.h"
#include "src/pthread/pthread_setspecific.h"
#include "test/IntegrationTest/test.h"

#include "hdr/pthread_macros.h"

// To make pthread_cleanup_push/pop work outside of LIBC_NAMESPACE
#define __pthread_cleanup_push LIBC_NAMESPACE::__pthread_cleanup_push
#define __pthread_cleanup_pop LIBC_NAMESPACE::__pthread_cleanup_pop

static int call_order = 0;
static void record_routine(void *arg) {
  *reinterpret_cast<int *>(arg) = ++call_order;
}

// 1. Test direct pop with execute = 0 and execute = 1 on the main thread.
static void test_direct_pop() {
  int val = 0;
  call_order = 0;
  pthread_cleanup_push(record_routine, &val);
  pthread_cleanup_pop(0);
  ASSERT_EQ(val, 0);

  pthread_cleanup_push(record_routine, &val);
  pthread_cleanup_pop(1);
  ASSERT_EQ(val, 1);
}

// 2. Test nested push / pop blocks in LIFO order on the main thread.
static void test_nested_pop() {
  int order[2] = {0, 0};
  call_order = 0;

  pthread_cleanup_push(record_routine, &order[1]);
  pthread_cleanup_push(record_routine, &order[0]);
  pthread_cleanup_pop(1);
  pthread_cleanup_pop(1);

  ASSERT_EQ(order[0], 1);
  ASSERT_EQ(order[1], 2);
}

// 3. Test sequential push / pop blocks in the same scope.
static void test_sequential_pop() {
  int first = 0;
  int second = 0;
  call_order = 0;

  pthread_cleanup_push(record_routine, &first);
  pthread_cleanup_pop(1);
  pthread_cleanup_push(record_routine, &second);
  pthread_cleanup_pop(1);

  ASSERT_EQ(first, 1);
  ASSERT_EQ(second, 2);
}

// 4. Test pthread_exit running handlers in LIFO order.
static int exit_order[3] = {0, 0, 0};

static void *thread_exit_lifo_func(void *) {
  pthread_cleanup_push(record_routine, &exit_order[2]);
  pthread_cleanup_push(record_routine, &exit_order[1]);
  pthread_cleanup_push(record_routine, &exit_order[0]);

  LIBC_NAMESPACE::pthread_exit(nullptr);

  pthread_cleanup_pop(0);
  pthread_cleanup_pop(0);
  pthread_cleanup_pop(0);
  return nullptr;
}

static void test_exit_lifo() {
  pthread_t th;
  void *retval = nullptr;

  call_order = 0;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_create(&th, nullptr, thread_exit_lifo_func,
                                           nullptr),
            0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_join(th, &retval), 0);

  ASSERT_EQ(exit_order[0], 1);
  ASSERT_EQ(exit_order[1], 2);
  ASSERT_EQ(exit_order[2], 3);
}

// 5. Test that handlers popped prior to pthread_exit are not executed.
static int exit_called1 = 0;
static int exit_called2 = 0;

static void *thread_popped_func(void *) {
  pthread_cleanup_push(record_routine, &exit_called1);
  pthread_cleanup_push(record_routine, &exit_called2);
  pthread_cleanup_pop(0); // Pop handler2 without executing

  LIBC_NAMESPACE::pthread_exit(nullptr);

  pthread_cleanup_pop(0);
  return nullptr;
}

static void test_exit_popped_not_called() {
  pthread_t th;
  void *retval = nullptr;

  call_order = 0;
  ASSERT_EQ(
      LIBC_NAMESPACE::pthread_create(&th, nullptr, thread_popped_func, nullptr),
      0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_join(th, &retval), 0);

  ASSERT_EQ(exit_called1, 1);
  ASSERT_EQ(exit_called2, 0);
}

// 6. Test that cleanup handlers run before TSS destructors.
static pthread_key_t tss_key;
static int global_sequence = 0;
static int cleanup_sequence = 0;
static int tss_sequence = 0;
static int dummy_tss_val = 99;

static void tss_destructor(void *) { tss_sequence = ++global_sequence; }

static void tss_cleanup_handler(void *) {
  ASSERT_EQ(LIBC_NAMESPACE::pthread_getspecific(tss_key), &dummy_tss_val);
  cleanup_sequence = ++global_sequence;
}

static void *thread_tss_order_func(void *) {
  LIBC_NAMESPACE::pthread_setspecific(tss_key, &dummy_tss_val);
  pthread_cleanup_push(tss_cleanup_handler, nullptr);

  LIBC_NAMESPACE::pthread_exit(nullptr);

  pthread_cleanup_pop(0);
  return nullptr;
}

static void test_exit_order_before_tss() {
  ASSERT_EQ(LIBC_NAMESPACE::pthread_key_create(&tss_key, tss_destructor), 0);

  pthread_t th;
  void *retval = nullptr;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_create(&th, nullptr, thread_tss_order_func,
                                           nullptr),
            0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_join(th, &retval), 0);

  ASSERT_EQ(cleanup_sequence, 1);
  ASSERT_EQ(tss_sequence, 2);

  ASSERT_EQ(LIBC_NAMESPACE::pthread_key_delete(tss_key), 0);
}

// 7. Test re-entrancy: a cleanup handler pushes and pops an inner handler.
static bool outer_called = false;
static bool inner_called = false;

static void inner_handler(void *) { inner_called = true; }

static void outer_handler(void *) {
  outer_called = true;
  pthread_cleanup_push(inner_handler, nullptr);
  pthread_cleanup_pop(1);
}

static void *thread_reentrancy_func(void *) {
  pthread_cleanup_push(outer_handler, nullptr);
  LIBC_NAMESPACE::pthread_exit(nullptr);

  pthread_cleanup_pop(0);
  return nullptr;
}

static void test_reentrancy() {
  pthread_t th;
  void *retval = nullptr;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_create(&th, nullptr, thread_reentrancy_func,
                                           nullptr),
            0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_join(th, &retval), 0);

  ASSERT_TRUE(outer_called);
  ASSERT_TRUE(inner_called);
}

// 8. Test releasing a mutex in a cleanup function.
static pthread_mutex_t test_mutex;

static void mutex_cleanup_routine(void *arg) {
  auto *mutex = reinterpret_cast<pthread_mutex_t *>(arg);
  LIBC_NAMESPACE::pthread_mutex_unlock(mutex);
}

static void *thread_mutex_cleanup_func(void *) {
  LIBC_NAMESPACE::pthread_mutex_lock(&test_mutex);
  pthread_cleanup_push(mutex_cleanup_routine, &test_mutex);

  // Thread exits while holding the mutex. The cleanup handler should unlock it.
  LIBC_NAMESPACE::pthread_exit(nullptr);

  pthread_cleanup_pop(0);
  return nullptr;
}

static void test_mutex_cleanup() {
  ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_init(&test_mutex, nullptr), 0);

  pthread_t th;
  void *retval = nullptr;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_create(&th, nullptr,
                                           thread_mutex_cleanup_func, nullptr),
            0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_join(th, &retval), 0);

  // Since the cleanup handler released the mutex, we should now be able to
  // lock it without blocking.
  ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_trylock(&test_mutex), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_unlock(&test_mutex), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_destroy(&test_mutex), 0);
}

TEST_MAIN() {
  test_direct_pop();
  test_nested_pop();
  test_sequential_pop();
  test_exit_lifo();
  test_exit_popped_not_called();
  test_exit_order_before_tss();
  test_reentrancy();
  test_mutex_cleanup();
  return 0;
}
