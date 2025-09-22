//===-- Tests for pthread_mutex_t -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/stdint_proxy.h" // uintptr_t
#include "src/pthread/pthread_create.h"
#include "src/pthread/pthread_join.h"
#include "src/pthread/pthread_mutex_destroy.h"
#include "src/pthread/pthread_mutex_init.h"
#include "src/pthread/pthread_mutex_lock.h"
#include "src/pthread/pthread_mutex_unlock.h"
#include "test/IntegrationTest/test.h"

#include <pthread.h>

constexpr int START = 0;
constexpr int MAX = 10000;

pthread_mutex_t mutex;
static int shared_int = START;

void *counter(void *arg) {
  int last_count = START;
  while (true) {
    LIBC_NAMESPACE::pthread_mutex_lock(&mutex);
    if (shared_int == last_count + 1) {
      shared_int++;
      last_count = shared_int;
    }
    LIBC_NAMESPACE::pthread_mutex_unlock(&mutex);
    if (last_count >= MAX)
      break;
  }
  return nullptr;
}

void relay_counter() {
  ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_init(&mutex, nullptr), 0);

  // The idea of this test is that two competing threads will update
  // a counter only if the other thread has updated it.
  pthread_t thread;
  LIBC_NAMESPACE::pthread_create(&thread, nullptr, counter, nullptr);

  int last_count = START;
  while (true) {
    ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_lock(&mutex), 0);
    if (shared_int == START) {
      ++shared_int;
      last_count = shared_int;
    } else if (shared_int != last_count) {
      ASSERT_EQ(shared_int, last_count + 1);
      ++shared_int;
      last_count = shared_int;
    }
    ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_unlock(&mutex), 0);
    if (last_count > MAX)
      break;
  }

  void *retval = reinterpret_cast<void *>(123);
  LIBC_NAMESPACE::pthread_join(thread, &retval);
  ASSERT_EQ(uintptr_t(retval), uintptr_t(nullptr));

  LIBC_NAMESPACE::pthread_mutex_destroy(&mutex);
}

pthread_mutex_t start_lock, step_lock;
bool started, step;

void *stepper(void *arg) {
  LIBC_NAMESPACE::pthread_mutex_lock(&start_lock);
  started = true;
  LIBC_NAMESPACE::pthread_mutex_unlock(&start_lock);

  LIBC_NAMESPACE::pthread_mutex_lock(&step_lock);
  step = true;
  LIBC_NAMESPACE::pthread_mutex_unlock(&step_lock);
  return nullptr;
}

void wait_and_step() {
  ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_init(&start_lock, nullptr), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_init(&step_lock, nullptr), 0);

  // In this test, we start a new thread but block it before it can make a
  // step. Once we ensure that the thread is blocked, we unblock it.
  // After unblocking, we then verify that the thread was indeed unblocked.
  step = false;
  started = false;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_lock(&step_lock), 0);

  pthread_t thread;
  LIBC_NAMESPACE::pthread_create(&thread, nullptr, stepper, nullptr);

  while (true) {
    // Make sure the thread actually started.
    ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_lock(&start_lock), 0);
    bool s = started;
    ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_unlock(&start_lock), 0);
    if (s)
      break;
  }

  // Since |step_lock| is still locked, |step| should be false.
  ASSERT_FALSE(step);

  // Unlock the step lock and wait until the step is made.
  ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_unlock(&step_lock), 0);

  while (true) {
    ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_lock(&step_lock), 0);
    bool current_step_value = step;
    ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_unlock(&step_lock), 0);
    if (current_step_value)
      break;
  }

  void *retval = reinterpret_cast<void *>(123);
  LIBC_NAMESPACE::pthread_join(thread, &retval);
  ASSERT_EQ(uintptr_t(retval), uintptr_t(nullptr));

  LIBC_NAMESPACE::pthread_mutex_destroy(&start_lock);
  LIBC_NAMESPACE::pthread_mutex_destroy(&step_lock);
}

static constexpr int THREAD_COUNT = 10;
static pthread_mutex_t multiple_waiter_lock;
static pthread_mutex_t counter_lock;
static int wait_count = 0;

void *waiter_func(void *) {
  LIBC_NAMESPACE::pthread_mutex_lock(&counter_lock);
  ++wait_count;
  LIBC_NAMESPACE::pthread_mutex_unlock(&counter_lock);

  // Block on the waiter lock until the main
  // thread unblocks.
  LIBC_NAMESPACE::pthread_mutex_lock(&multiple_waiter_lock);
  LIBC_NAMESPACE::pthread_mutex_unlock(&multiple_waiter_lock);

  LIBC_NAMESPACE::pthread_mutex_lock(&counter_lock);
  --wait_count;
  LIBC_NAMESPACE::pthread_mutex_unlock(&counter_lock);

  return nullptr;
}

void multiple_waiters() {
  LIBC_NAMESPACE::pthread_mutex_init(&multiple_waiter_lock, nullptr);
  LIBC_NAMESPACE::pthread_mutex_init(&counter_lock, nullptr);

  LIBC_NAMESPACE::pthread_mutex_lock(&multiple_waiter_lock);
  pthread_t waiters[THREAD_COUNT];
  for (int i = 0; i < THREAD_COUNT; ++i) {
    LIBC_NAMESPACE::pthread_create(waiters + i, nullptr, waiter_func, nullptr);
  }

  // Spin until the counter is incremented to the desired
  // value.
  while (true) {
    LIBC_NAMESPACE::pthread_mutex_lock(&counter_lock);
    if (wait_count == THREAD_COUNT) {
      LIBC_NAMESPACE::pthread_mutex_unlock(&counter_lock);
      break;
    }
    LIBC_NAMESPACE::pthread_mutex_unlock(&counter_lock);
  }

  LIBC_NAMESPACE::pthread_mutex_unlock(&multiple_waiter_lock);

  void *retval;
  for (int i = 0; i < THREAD_COUNT; ++i) {
    LIBC_NAMESPACE::pthread_join(waiters[i], &retval);
  }

  ASSERT_EQ(wait_count, 0);

  LIBC_NAMESPACE::pthread_mutex_destroy(&multiple_waiter_lock);
  LIBC_NAMESPACE::pthread_mutex_destroy(&counter_lock);
}

// Test the initializer
[[maybe_unused]]
static pthread_mutex_t test_initializer = PTHREAD_MUTEX_INITIALIZER;

TEST_MAIN() {
  relay_counter();
  wait_and_step();
  multiple_waiters();
  return 0;
}
