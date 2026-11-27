//===-- Tests for pthread_join-- ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/pthread/pthread_create.h"
#include "src/pthread/pthread_join.h"
#include "src/pthread/pthread_self.h"

#include "src/__support/CPP/atomic.h"
#include "src/__support/threads/thread.h"
#include "test/IntegrationTest/test.h"

#include <errno.h>
#include <pthread.h>

static void *simple_func(void *) { return nullptr; }

static void null_join_test() {
  pthread_t tid;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_create(&tid, nullptr, simple_func, nullptr),
            0);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(LIBC_NAMESPACE::pthread_join(tid, nullptr), 0);
  ASSERT_ERRNO_SUCCESS();
}

static void self_join_test() {
  ASSERT_EQ(
      LIBC_NAMESPACE::pthread_join(LIBC_NAMESPACE::pthread_self(), nullptr),
      EDEADLK);
}

struct MutualJoinArgs {
  pthread_t *peer;
  LIBC_NAMESPACE::cpp::Atomic<int> *ready_count;
  LIBC_NAMESPACE::cpp::Atomic<int> *start;
  LIBC_NAMESPACE::cpp::Atomic<int> *result;
  int start_value;
};

static void *mutual_join_func(void *arg) {
  auto *args = reinterpret_cast<MutualJoinArgs *>(arg);
  args->ready_count->fetch_add(1);
  while (args->start->load() < args->start_value)
    ; // Spin until this thread is released to join its peer.

  args->result->store(LIBC_NAMESPACE::pthread_join(*args->peer, nullptr));
  return nullptr;
}

static bool is_joining(pthread_t joiner, pthread_t target) {
  auto *joiner_thread = reinterpret_cast<LIBC_NAMESPACE::Thread *>(&joiner);
  auto *target_thread = reinterpret_cast<LIBC_NAMESPACE::Thread *>(&target);
  return target_thread->attrib->joiner.load() == joiner_thread->attrib;
}

static void mutual_join_test() {
  pthread_t thread1;
  pthread_t thread2;
  LIBC_NAMESPACE::cpp::Atomic<int> ready_count(0);
  LIBC_NAMESPACE::cpp::Atomic<int> start(0);
  LIBC_NAMESPACE::cpp::Atomic<int> result1(-1);
  LIBC_NAMESPACE::cpp::Atomic<int> result2(-1);

  MutualJoinArgs args1{&thread2, &ready_count, &start, &result1, 1};
  MutualJoinArgs args2{&thread1, &ready_count, &start, &result2, 2};

  ASSERT_EQ(LIBC_NAMESPACE::pthread_create(&thread1, nullptr, mutual_join_func,
                                           &args1),
            0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_create(&thread2, nullptr, mutual_join_func,
                                           &args2),
            0);

  while (ready_count.load() != 2)
    ; // Spin until both threads are ready to join each other.
  start.store(1);
  while (!is_joining(thread1, thread2))
    ; // Spin until thread1 has started joining thread2.
  start.store(2);

  while (result1.load() == -1 || result2.load() == -1)
    ; // Spin until the successful joiner and deadlock loser have both exited.

  // A thread is recovered to joinable state if its joining requester gets
  // EDEADLK.
  bool thread1_joinable = result2.load() == EDEADLK;
  bool thread2_joinable = result1.load() == EDEADLK;

  ASSERT_TRUE(thread1_joinable || thread2_joinable);

  if (thread1_joinable)
    ASSERT_EQ(LIBC_NAMESPACE::pthread_join(thread1, nullptr), 0);
  if (thread2_joinable)
    ASSERT_EQ(LIBC_NAMESPACE::pthread_join(thread2, nullptr), 0);
}

TEST_MAIN() {
  errno = 0;
  null_join_test();
  self_join_test();
  mutual_join_test();
  return 0;
}
