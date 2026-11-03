//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Integration tests for pthread_kill.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/signal_macros.h"
#include "hdr/stdint_proxy.h"
#include "hdr/types/pthread_t.h"
#include "hdr/types/sigset_t.h"
#include "hdr/types/struct_sigaction.h"
#include "src/__support/CPP/atomic.h"
#include "src/__support/threads/futex_utils.h"
#include "src/__support/threads/thread.h"
#include "src/pthread/pthread_create.h"
#include "src/pthread/pthread_join.h"
#include "src/pthread/pthread_self.h"
#include "src/signal/pthread_kill.h"
#include "src/signal/pthread_sigmask.h"
#include "src/signal/sigaction.h"
#include "src/signal/sigaddset.h"
#include "src/signal/sigemptyset.h"
#include "test/IntegrationTest/test.h"

static LIBC_NAMESPACE::Futex usr1_count(0);
static LIBC_NAMESPACE::Futex usr2_count(0);

static void sigusr1_handler(int) {
  usr1_count.fetch_add(1, LIBC_NAMESPACE::cpp::MemoryOrder::RELAXED);
}

static void sigusr2_handler(int) {
  usr2_count.fetch_add(1, LIBC_NAMESPACE::cpp::MemoryOrder::RELAXED);
}

static void setup_signal_handlers() {
  struct sigaction sa = {};
  sa.sa_handler = sigusr1_handler;
  sa.sa_flags = 0;
  LIBC_NAMESPACE::sigemptyset(&sa.sa_mask);
  ASSERT_EQ(LIBC_NAMESPACE::sigaction(SIGUSR1, &sa, nullptr), 0);

  sa.sa_handler = sigusr2_handler;
  ASSERT_EQ(LIBC_NAMESPACE::sigaction(SIGUSR2, &sa, nullptr), 0);
}

static void test_invalid_signal() {
  pthread_t self = LIBC_NAMESPACE::pthread_self();
  ASSERT_EQ(LIBC_NAMESPACE::pthread_kill(self, -1), EINVAL);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_kill(self, 1000), EINVAL);
}

static void test_self_signal_zero() {
  pthread_t self = LIBC_NAMESPACE::pthread_self();
  ASSERT_EQ(LIBC_NAMESPACE::pthread_kill(self, 0), 0);
}

static void test_self_signal_delivery() {
  pthread_t self = LIBC_NAMESPACE::pthread_self();
  uint32_t initial = usr1_count.load(LIBC_NAMESPACE::cpp::MemoryOrder::RELAXED);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_kill(self, SIGUSR1), 0);
  ASSERT_EQ(usr1_count.load(LIBC_NAMESPACE::cpp::MemoryOrder::RELAXED),
            initial + 1);
}

static void *cross_thread_worker(void *initial_opaque) {
  uint32_t initial =
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(initial_opaque));
  usr1_count.wait(initial);
  return nullptr;
}

static void test_cross_thread_signal() {
  pthread_t th;
  uint32_t initial = usr1_count.load(LIBC_NAMESPACE::cpp::MemoryOrder::RELAXED);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_create(&th, nullptr, cross_thread_worker,
                                           reinterpret_cast<void *>(initial)),
            0);

  ASSERT_EQ(LIBC_NAMESPACE::pthread_kill(th, 0), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_kill(th, SIGUSR1), 0);

  void *retval;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_join(th, &retval), 0);
}

static void *zombie_worker(void *) { return nullptr; }

static void test_zombie_thread() {
  pthread_t th;
  ASSERT_EQ(
      LIBC_NAMESPACE::pthread_create(&th, nullptr, zombie_worker, nullptr), 0);

  // Wait until thread has exited.
  auto *thread_internal = reinterpret_cast<LIBC_NAMESPACE::Thread *>(&th);
  thread_internal->wait();

  // POSIX.1-2024 requires that a zombie thread ID does not return ESRCH.
  ASSERT_EQ(LIBC_NAMESPACE::pthread_kill(th, 0), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_kill(th, -1), EINVAL);

  void *retval;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_join(th, &retval), 0);
}

static LIBC_NAMESPACE::Futex mask_ready(0);
static LIBC_NAMESPACE::Futex unblock_signal(0);

static void *mask_worker(void *) {
  sigset_t set;
  LIBC_NAMESPACE::sigemptyset(&set);
  LIBC_NAMESPACE::sigaddset(&set, SIGUSR2);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_sigmask(SIG_BLOCK, &set, nullptr), 0);

  mask_ready.store(1, LIBC_NAMESPACE::cpp::MemoryOrder::RELEASE);
  mask_ready.notify_one();

  unblock_signal.wait(0);

  // Signal was sent while blocked. It should not have been delivered yet.
  ASSERT_EQ(usr2_count.load(LIBC_NAMESPACE::cpp::MemoryOrder::RELAXED), 0);

  // Unblocking the signal should cause immediate delivery.
  ASSERT_EQ(LIBC_NAMESPACE::pthread_sigmask(SIG_UNBLOCK, &set, nullptr), 0);

  ASSERT_EQ(usr2_count.load(LIBC_NAMESPACE::cpp::MemoryOrder::RELAXED), 1);
  return nullptr;
}

static void test_signal_mask_interaction() {
  mask_ready.store(0, LIBC_NAMESPACE::cpp::MemoryOrder::RELAXED);
  unblock_signal.store(0, LIBC_NAMESPACE::cpp::MemoryOrder::RELAXED);
  usr2_count.store(0, LIBC_NAMESPACE::cpp::MemoryOrder::RELAXED);

  pthread_t th;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_create(&th, nullptr, mask_worker, nullptr),
            0);

  mask_ready.wait(0);

  ASSERT_EQ(LIBC_NAMESPACE::pthread_kill(th, SIGUSR2), 0);

  unblock_signal.store(1, LIBC_NAMESPACE::cpp::MemoryOrder::RELEASE);
  unblock_signal.notify_one();

  void *retval;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_join(th, &retval), 0);
  ASSERT_EQ(usr2_count.load(LIBC_NAMESPACE::cpp::MemoryOrder::RELAXED), 1);
}

TEST_MAIN() {
  setup_signal_handlers();

  test_invalid_signal();
  test_self_signal_zero();
  test_self_signal_delivery();
  test_cross_thread_signal();
  test_zombie_thread();
  test_signal_mask_interaction();

  return 0;
}
