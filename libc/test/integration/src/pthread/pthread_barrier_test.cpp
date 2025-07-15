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
#include "src/stdio/printf.h"

#include "test/IntegrationTest/test.h"

#include <pthread.h>

pthread_barrier_t barrier;

void smoke_test() {
  ASSERT_EQ(LIBC_NAMESPACE::pthread_barrier_init(&barrier, nullptr, 1), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_barrier_wait(&barrier),
            PTHREAD_BARRIER_SERIAL_THREAD);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_barrier_destroy(&barrier), 0);
}

LIBC_NAMESPACE::cpp::Atomic<int> counter;
void *increment_counter_and_wait(void *args) {
  counter.fetch_add(1);
  LIBC_NAMESPACE::pthread_barrier_wait(&barrier);
  return 0;
}

void single_use_barrier() {
  counter.set(0);
  const int NUM_THREADS = 30;
  pthread_t threads[NUM_THREADS];
  ASSERT_EQ(
      LIBC_NAMESPACE::pthread_barrier_init(&barrier, nullptr, NUM_THREADS + 1),
      0);

  for (int i = 0; i < NUM_THREADS; ++i)
    LIBC_NAMESPACE::pthread_create(&threads[i], nullptr,
                                   increment_counter_and_wait, nullptr);

  LIBC_NAMESPACE::pthread_barrier_wait(&barrier);
  ASSERT_EQ(counter.load(), NUM_THREADS);

  for (int i = 0; i < NUM_THREADS; ++i)
    LIBC_NAMESPACE::pthread_join(threads[i], nullptr);

  LIBC_NAMESPACE::pthread_barrier_destroy(&barrier);
}

void reusable_barrier() {
  counter.set(0);
  const int NUM_THREADS = 30;
  const int REPEAT = 20;
  pthread_t threads[NUM_THREADS * REPEAT];
  ASSERT_EQ(
      LIBC_NAMESPACE::pthread_barrier_init(&barrier, nullptr, NUM_THREADS + 1),
      0);

  for (int i = 0; i < REPEAT; ++i) {
    for (int j = 0; j < NUM_THREADS; ++j)
      LIBC_NAMESPACE::pthread_create(&threads[NUM_THREADS * i + j], nullptr,
                                     increment_counter_and_wait, nullptr);

    LIBC_NAMESPACE::pthread_barrier_wait(&barrier);
    ASSERT_EQ(counter.load(), NUM_THREADS * (i + 1));
  }

  for (int i = 0; i < NUM_THREADS * REPEAT; ++i)
    LIBC_NAMESPACE::pthread_join(threads[i], nullptr);

  LIBC_NAMESPACE::pthread_barrier_destroy(&barrier);
}

void *barrier_wait(void* in) {
  return reinterpret_cast<void *>(
      LIBC_NAMESPACE::pthread_barrier_wait(&barrier));
}

// verify that only one of the wait() calls return PTHREAD_BARRIER_SERIAL_THREAD
// with the rest returning 0
void one_nonzero_wait_returnval() {
  const int NUM_THREADS = 30;
  pthread_t threads[NUM_THREADS];
  LIBC_NAMESPACE::pthread_barrier_init(&barrier, nullptr, NUM_THREADS + 1);
  for (int i = 0; i < NUM_THREADS; ++i)
    LIBC_NAMESPACE::pthread_create(&threads[i], nullptr, barrier_wait, nullptr);

  uintptr_t retsum = LIBC_NAMESPACE::pthread_barrier_wait(&barrier);
  for (int i = 0; i < NUM_THREADS; ++i) {
    void* ret;
    LIBC_NAMESPACE::pthread_join(threads[i], &ret);
    retsum += reinterpret_cast<uintptr_t>(ret);
  }

  ASSERT_EQ(static_cast<int>(retsum), PTHREAD_BARRIER_SERIAL_THREAD);
}

TEST_MAIN() {
  smoke_test();
  single_use_barrier();
  reusable_barrier();
  one_nonzero_wait_returnval();
  return 0;
}
