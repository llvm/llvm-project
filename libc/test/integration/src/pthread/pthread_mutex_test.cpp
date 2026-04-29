//===-- Tests for pthread_mutex_t -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/stdint_proxy.h" // uintptr_t
#include "src/pthread/pthread_create.h"
#include "src/pthread/pthread_join.h"
#include "src/pthread/pthread_mutex_destroy.h"
#include "src/pthread/pthread_mutex_init.h"
#include "src/pthread/pthread_mutex_lock.h"
#include "src/pthread/pthread_mutex_trylock.h"
#include "src/pthread/pthread_mutex_unlock.h"
#include "src/pthread/pthread_mutexattr_destroy.h"
#include "src/pthread/pthread_mutexattr_init.h"
#include "src/pthread/pthread_mutexattr_settype.h"
#include "src/string/memory_utils/inline_memcpy.h"
#include "test/IntegrationTest/test.h"

#include <pthread.h>

constexpr int START = 0;
constexpr int MAX = 10000;

static pthread_mutex_t snapshot_mutex(const void *mutex_storage) {
  pthread_mutex_t snapshot;
  // The original storage may currently hold libc's internal mutex
  // representation. Copy the bytes into pthread_mutex_t storage before
  // inspection to avoid strict aliasing violations.
  LIBC_NAMESPACE::inline_memcpy(&snapshot, mutex_storage, sizeof(snapshot));
  return snapshot;
}

pthread_mutex_t mutex;
static int shared_int = START;

void *counter([[maybe_unused]] void *arg) {
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

void *stepper([[maybe_unused]] void *arg) {
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

void trylock_test() {
  pthread_mutex_t trylock_mutex;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_init(&trylock_mutex, nullptr), 0);

  ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_trylock(&trylock_mutex), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_trylock(&trylock_mutex), EBUSY);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_unlock(&trylock_mutex), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_trylock(&trylock_mutex), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_unlock(&trylock_mutex), 0);

  LIBC_NAMESPACE::pthread_mutex_destroy(&trylock_mutex);
}

void *trylock_other_thread(void *arg) {
  auto *mutex = reinterpret_cast<pthread_mutex_t *>(arg);
  int result = LIBC_NAMESPACE::pthread_mutex_trylock(mutex);
  if (result == 0)
    ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_unlock(mutex), 0);
  return reinterpret_cast<void *>(uintptr_t(result));
}

void recursive_mutex_test() {
  pthread_mutexattr_t attr;
  pthread_mutex_t recursive_mutex;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_mutexattr_init(&attr), 0);
  ASSERT_EQ(
      LIBC_NAMESPACE::pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE),
      0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_init(&recursive_mutex, &attr), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_mutexattr_destroy(&attr), 0);

  pthread_mutex_t snapshot = snapshot_mutex(&recursive_mutex);
  ASSERT_TRUE(snapshot.__recursive);
  ASSERT_EQ(snapshot.__owner, 0);
  ASSERT_EQ(snapshot.__lock_count, size_t(0));

  ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_lock(&recursive_mutex), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_lock(&recursive_mutex), 0);

  pthread_t thread;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_create(
                &thread, nullptr, trylock_other_thread, &recursive_mutex),
            0);
  void *retval = nullptr;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_join(thread, &retval), 0);
  ASSERT_EQ(uintptr_t(retval), uintptr_t(EBUSY));

  ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_unlock(&recursive_mutex), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_unlock(&recursive_mutex), 0);
  snapshot = snapshot_mutex(&recursive_mutex);
  ASSERT_EQ(snapshot.__owner, 0);
  ASSERT_EQ(snapshot.__lock_count, size_t(0));

  ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_trylock(&recursive_mutex), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_unlock(&recursive_mutex), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_destroy(&recursive_mutex), 0);
}

[[maybe_unused]]
static pthread_mutex_t test_initializer = PTHREAD_MUTEX_INITIALIZER;

// POSIX.1 requires PTHREAD_MUTEX_INITIALIZER is consistent with
// pthread_mutex_init(m, nullptr).
void initializer_acts_the_same_as_null_attr() {
  pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
  pthread_mutex_t mutex_from_init;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_init(&mutex_from_init, nullptr), 0);

  pthread_mutex_t mutex_snapshot = snapshot_mutex(&mutex);
  pthread_mutex_t mutex_from_init_snapshot = snapshot_mutex(&mutex_from_init);
  // Do per-field comparison. We cannot do direct bytewise comparison because
  // the layout has padding bits and __builtin_clear_padding is not available.
  ASSERT_EQ(mutex_snapshot.__ftxw.__word,
            mutex_from_init_snapshot.__ftxw.__word);
  ASSERT_EQ(mutex_snapshot.__priority_inherit,
            mutex_from_init_snapshot.__priority_inherit);
  ASSERT_EQ(mutex_snapshot.__recursive, mutex_from_init_snapshot.__recursive);
  ASSERT_EQ(mutex_snapshot.__robust, mutex_from_init_snapshot.__robust);
  ASSERT_EQ(mutex_snapshot.__pshared, mutex_from_init_snapshot.__pshared);
  ASSERT_EQ(mutex_snapshot.__owner, mutex_from_init_snapshot.__owner);
  ASSERT_EQ(mutex_snapshot.__lock_count, mutex_from_init_snapshot.__lock_count);

  ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_lock(&mutex), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_trylock(&mutex), EBUSY);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_unlock(&mutex), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_destroy(&mutex), 0);

  ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_lock(&mutex_from_init), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_trylock(&mutex_from_init), EBUSY);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_unlock(&mutex_from_init), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_destroy(&mutex_from_init), 0);
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

TEST_MAIN() {
  relay_counter();
  wait_and_step();
  trylock_test();
  recursive_mutex_test();
  initializer_acts_the_same_as_null_attr();
  multiple_waiters();
  return 0;
}
