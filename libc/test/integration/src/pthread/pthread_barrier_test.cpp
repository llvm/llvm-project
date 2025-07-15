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
#include <stdint.h> // uintptr_t

pthread_barrier_t barrier;

void smoke_test() {
  ASSERT_EQ(LIBC_NAMESPACE::pthread_barrier_init(&barrier, nullptr, 1), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_barrier_wait(&barrier), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_barrier_destroy(&barrier), 0);
}

LIBC_NAMESPACE::cpp::Atomic<int> counter;
void *increment_counter_and_wait(void *args) {
  counter.fetch_add(1);
  LIBC_NAMESPACE::pthread_barrier_wait(&barrier);
  return 0;
}

void shared_counter() {
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
}

void reusable_shared_counter() {
  counter.set(0);
  const int NUM_THREADS = 30;
  const int REPEAT = 10;
  pthread_t threads[NUM_THREADS * REPEAT];
  ASSERT_EQ(
      LIBC_NAMESPACE::pthread_barrier_init(&barrier, nullptr, NUM_THREADS + 1),
      0);

  for (int i = 0; i < REPEAT; ++i) {
    for (int j = 0; j < NUM_THREADS; ++j) {
      LIBC_NAMESPACE::pthread_create(&threads[NUM_THREADS * i + j], nullptr,
                                     increment_counter_and_wait, nullptr);
    }
    LIBC_NAMESPACE::pthread_barrier_wait(&barrier);
    ASSERT_EQ(counter.load(), NUM_THREADS * (i + 1));
  }

  for (int i = 0; i < NUM_THREADS * REPEAT; ++i) {
    LIBC_NAMESPACE::pthread_join(threads[i], nullptr);
  }
}

TEST_MAIN() {
  smoke_test();
  shared_counter();
  reusable_shared_counter();
  return 0;
}
