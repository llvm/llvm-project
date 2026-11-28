//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Integration tests for pthread_getcpuclockid.
///
//===----------------------------------------------------------------------===//

#include "hdr/time_macros.h"
#include "hdr/types/clockid_t.h"
#include "hdr/types/pthread_t.h"
#include "hdr/types/struct_timespec.h"
#include "hdr/types/time_t.h"
#include "src/__support/CPP/atomic.h"
#include "src/__support/threads/futex_utils.h"
#include "src/__support/threads/thread.h"
#include "src/pthread/pthread_create.h"
#include "src/pthread/pthread_getcpuclockid.h"
#include "src/pthread/pthread_join.h"
#include "src/pthread/pthread_self.h"
#include "src/time/clock_gettime.h"
#include "test/IntegrationTest/test.h"

// How much CPU time the busy worker accumulates before reporting back.
static constexpr time_t BUSY_TIME_NS = 10'000'000;

static LIBC_NAMESPACE::Futex worker_busy_done(0);
static LIBC_NAMESPACE::Futex worker_release(0);

static time_t cpu_time_ns(clockid_t clock) {
  timespec ts;
  ASSERT_EQ(LIBC_NAMESPACE::clock_gettime(clock, &ts), 0);
  return ts.tv_sec * 1'000'000'000 + ts.tv_nsec;
}

static void test_self() {
  clockid_t clock;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_getcpuclockid(
                LIBC_NAMESPACE::pthread_self(), &clock),
            0);
  // The clock must identify this specific thread, not "the calling thread".
  ASSERT_NE(clock, CLOCK_THREAD_CPUTIME_ID);

  // Both clocks measure the CPU time of this thread, so the first reading
  // cannot be larger than the second one.
  time_t first = cpu_time_ns(clock);
  time_t second = cpu_time_ns(CLOCK_THREAD_CPUTIME_ID);
  ASSERT_TRUE(first <= second);
}

static void *busy_worker(void *) {
  while (cpu_time_ns(CLOCK_THREAD_CPUTIME_ID) < BUSY_TIME_NS)
    ;
  worker_busy_done.store(1, LIBC_NAMESPACE::cpp::MemoryOrder::RELEASE);
  worker_busy_done.notify_one();

  // Stay alive until the main thread is done reading our clock.
  while (worker_release.load(LIBC_NAMESPACE::cpp::MemoryOrder::ACQUIRE) == 0)
    worker_release.wait(0);
  return nullptr;
}

static void test_running_thread() {
  pthread_t th;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_create(&th, nullptr, busy_worker, nullptr),
            0);

  clockid_t worker_clock;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_getcpuclockid(th, &worker_clock), 0);
  clockid_t self_clock;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_getcpuclockid(
                LIBC_NAMESPACE::pthread_self(), &self_clock),
            0);
  ASSERT_NE(worker_clock, self_clock);

  while (worker_busy_done.load(LIBC_NAMESPACE::cpp::MemoryOrder::ACQUIRE) == 0)
    worker_busy_done.wait(0);
  // Reading the worker's clock from this thread must show the CPU time it
  // has accumulated.
  ASSERT_TRUE(cpu_time_ns(worker_clock) >= BUSY_TIME_NS);

  worker_release.store(1, LIBC_NAMESPACE::cpp::MemoryOrder::RELEASE);
  worker_release.notify_one();
  ASSERT_EQ(LIBC_NAMESPACE::pthread_join(th, nullptr), 0);
}

static LIBC_NAMESPACE::Futex exit_release(0);

static void *exiting_worker(void *) {
  // Don't exit until the main thread has obtained our clock ID.
  while (exit_release.load(LIBC_NAMESPACE::cpp::MemoryOrder::ACQUIRE) == 0)
    exit_release.wait(0);
  return nullptr;
}

static void test_exited_thread() {
  pthread_t th;
  ASSERT_EQ(
      LIBC_NAMESPACE::pthread_create(&th, nullptr, exiting_worker, nullptr), 0);
  clockid_t clock_before;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_getcpuclockid(th, &clock_before), 0);

  // Let the thread exit and wait until it has done so, but don't join it yet.
  exit_release.store(1, LIBC_NAMESPACE::cpp::MemoryOrder::RELEASE);
  exit_release.notify_one();
  auto *thread_internal = reinterpret_cast<LIBC_NAMESPACE::Thread *>(&th);
  thread_internal->wait();

  // The pthread_t remains valid until the thread is joined.
  clockid_t clock_after;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_getcpuclockid(th, &clock_after), 0);
  ASSERT_EQ(clock_before, clock_after);

  ASSERT_EQ(LIBC_NAMESPACE::pthread_join(th, nullptr), 0);
}

TEST_MAIN() {
  test_self();
  test_running_thread();
  test_exited_thread();
  return 0;
}
