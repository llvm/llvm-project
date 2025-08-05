//===-- Tests for pthread_barrier_t ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/pthread/pthread_barrier_destroy.h"
#include "src/pthread/pthread_barrier_init.h"
#include "src/pthread/pthread_barrier_wait.h"

#include "src/__support/CPP/atomic.h"
#include "src/pthread/pthread_create.h"
#include "src/pthread/pthread_join.h"
#include "src/pthread/pthread_mutex_destroy.h"
#include "src/pthread/pthread_mutex_init.h"
#include "src/pthread/pthread_mutex_lock.h"
#include "src/pthread/pthread_mutex_unlock.h"
#include "src/string/memset.h"

#include "test/IntegrationTest/test.h"

#include <pthread.h>

pthread_barrier_t barrier;
LIBC_NAMESPACE::cpp::Atomic<int> counter;

void *increment_counter_and_wait(void *args) {
  counter.fetch_add(1);
  return reinterpret_cast<void *>(
      LIBC_NAMESPACE::pthread_barrier_wait(&barrier));
}

void single_use_barrier_test(int num_threads) {
  counter.set(0);
  // create n - 1 ADDITIONAL threads since the current thread will also wait at
  // the barrier
  pthread_t threads[num_threads - 1];
  LIBC_NAMESPACE::memset(&barrier, 0, sizeof(pthread_barrier_t));
  ASSERT_EQ(
      LIBC_NAMESPACE::pthread_barrier_init(&barrier, nullptr, num_threads), 0);

  for (int i = 0; i < num_threads - 1; ++i)
    LIBC_NAMESPACE::pthread_create(&threads[i], nullptr,
                                   increment_counter_and_wait, nullptr);

  uintptr_t return_val_sum =
      reinterpret_cast<uintptr_t>(increment_counter_and_wait(nullptr));
  ASSERT_EQ(counter.load(), num_threads);

  // verify only one thread got the PTHREAD_BARRIER_SERIAL_THREAD return value
  for (int i = 0; i < num_threads - 1; ++i) {
    void *ret;
    LIBC_NAMESPACE::pthread_join(threads[i], &ret);
    if (reinterpret_cast<uintptr_t>(ret) ==
        static_cast<uintptr_t>(PTHREAD_BARRIER_SERIAL_THREAD)) {
      return_val_sum += reinterpret_cast<uintptr_t>(ret);
    } else {
      ASSERT_EQ(ret, 0);
    }
  }
  ASSERT_EQ(return_val_sum,
            static_cast<uintptr_t>(PTHREAD_BARRIER_SERIAL_THREAD));

  LIBC_NAMESPACE::pthread_barrier_destroy(&barrier);
}

void reused_barrier_test() {
  counter.set(0);
  const int NUM_THREADS = 30;
  const int REPEAT = 20;
  pthread_t threads[NUM_THREADS - 1]; // subtract 1 for main thread
  LIBC_NAMESPACE::memset(&barrier, 0, sizeof(pthread_barrier_t));
  ASSERT_EQ(
      LIBC_NAMESPACE::pthread_barrier_init(&barrier, nullptr, NUM_THREADS), 0);

  for (int i = 0; i < REPEAT; ++i) {
    for (int j = 0; j < NUM_THREADS - 1; ++j)
      LIBC_NAMESPACE::pthread_create(&threads[j], nullptr,
                                     increment_counter_and_wait, nullptr);

    uintptr_t return_val_sum =
        reinterpret_cast<uintptr_t>(increment_counter_and_wait(nullptr));
    ASSERT_EQ(counter.load(), NUM_THREADS * (i + 1));

    // verify only one thread got the PTHREAD_BARRIER_SERIAL_THREAD return value
    for (int i = 0; i < NUM_THREADS - 1; ++i) {
      void *ret;
      LIBC_NAMESPACE::pthread_join(threads[i], &ret);
      if (reinterpret_cast<uintptr_t>(ret) ==
          static_cast<uintptr_t>(PTHREAD_BARRIER_SERIAL_THREAD)) {
        return_val_sum += reinterpret_cast<uintptr_t>(ret);
      } else {
        ASSERT_EQ(ret, 0);
      }
    }
    ASSERT_EQ(return_val_sum,
              static_cast<uintptr_t>(PTHREAD_BARRIER_SERIAL_THREAD));
  }

  LIBC_NAMESPACE::pthread_barrier_destroy(&barrier);
}

void *barrier_wait(void *in) {
  return reinterpret_cast<void *>(
      LIBC_NAMESPACE::pthread_barrier_wait(&barrier));
}

TEST_MAIN() {
  // don't create any additional threads; only use main thread
  single_use_barrier_test(1);

  single_use_barrier_test(30);
  reused_barrier_test();
  return 0;
}
