//===-- sanitizer_stoptheworld_test.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for sanitizer_stoptheworld.h
//
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_stoptheworld.h"

#include "sanitizer_common/sanitizer_platform.h"
#if SANITIZER_LINUX && defined(__x86_64__)

#  include <mutex>
#  include <thread>

#  include "gtest/gtest.h"
#  include "sanitizer_common/sanitizer_common.h"
#  include "sanitizer_common/sanitizer_libc.h"

namespace __sanitizer {

static std::mutex incrementer_thread_exit_mutex;

struct CallbackArgument {
  volatile int counter;
  volatile bool threads_stopped;
  volatile bool callback_executed;
  CallbackArgument()
      : counter(0), threads_stopped(false), callback_executed(false) {}
};

void IncrementerThread(CallbackArgument &callback_argument) {
  while (true) {
    __sync_fetch_and_add(&callback_argument.counter, 1);

    if (incrementer_thread_exit_mutex.try_lock()) {
      incrementer_thread_exit_mutex.unlock();
      return;
    }

    std::this_thread::yield();
  }
}

// This callback checks that IncrementerThread is suspended at the time of its
// execution.
void Callback(const SuspendedThreadsList &suspended_threads_list,
              void *argument) {
  CallbackArgument *callback_argument = (CallbackArgument *)argument;
  callback_argument->callback_executed = true;
  int counter_at_init = __sync_fetch_and_add(&callback_argument->counter, 0);
  for (uptr i = 0; i < 1000; i++) {
    std::this_thread::yield();
    if (__sync_fetch_and_add(&callback_argument->counter, 0) !=
        counter_at_init) {
      callback_argument->threads_stopped = false;
      return;
    }
  }
  callback_argument->threads_stopped = true;
}

TEST(StopTheWorld, SuspendThreadsSimple) {
  CallbackArgument argument;
  std::thread thread;
  incrementer_thread_exit_mutex.lock();
  ASSERT_NO_THROW(thread = std::thread(IncrementerThread, std::ref(argument)));
  StopTheWorld(&Callback, &argument);
  incrementer_thread_exit_mutex.unlock();
  EXPECT_TRUE(argument.callback_executed);
  EXPECT_TRUE(argument.threads_stopped);
  // argument is on stack, so we have to wait for the incrementer thread to
  // terminate before we can return from this function.
  ASSERT_NO_THROW(thread.join());
}

// A more comprehensive test where we spawn a bunch of threads while executing
// StopTheWorld in parallel.
static const uptr kThreadCount = 50;
static const uptr kStopWorldAfter = 10;  // let this many threads spawn first

static std::mutex advanced_incrementer_thread_exit_mutex;

struct AdvancedCallbackArgument {
  volatile uptr thread_index;
  volatile int counters[kThreadCount];
  std::thread threads[kThreadCount];
  volatile bool threads_stopped;
  volatile bool callback_executed;
  volatile bool fatal_error;
  AdvancedCallbackArgument()
      : thread_index(0),
        threads_stopped(false),
        callback_executed(false),
        fatal_error(false) {}
};

void AdvancedIncrementerThread(AdvancedCallbackArgument &callback_argument) {
  uptr this_thread_index =
      __sync_fetch_and_add(&callback_argument.thread_index, 1);
  // Spawn the next thread.
  if (this_thread_index + 1 < kThreadCount) {
    try {
      callback_argument.threads[this_thread_index + 1] =
          std::thread(AdvancedIncrementerThread, std::ref(callback_argument));
    } catch (...) {
      // Cannot use ASSERT_EQ in non-void-returning functions. If there's a
      // problem, defer failing to the main thread.
      callback_argument.fatal_error = true;
      __sync_fetch_and_add(&callback_argument.thread_index,
                           kThreadCount - callback_argument.thread_index);
    }
  }
  // Do the actual work.
  while (true) {
    __sync_fetch_and_add(&callback_argument.counters[this_thread_index], 1);
    if (advanced_incrementer_thread_exit_mutex.try_lock()) {
      advanced_incrementer_thread_exit_mutex.unlock();
      return;
    }

    std::this_thread::yield();
  }
}

void AdvancedCallback(const SuspendedThreadsList &suspended_threads_list,
                      void *argument) {
  AdvancedCallbackArgument *callback_argument =
      (AdvancedCallbackArgument *)argument;
  callback_argument->callback_executed = true;

  int counters_at_init[kThreadCount];
  for (uptr j = 0; j < kThreadCount; j++)
    counters_at_init[j] =
        __sync_fetch_and_add(&callback_argument->counters[j], 0);
  for (uptr i = 0; i < 10; i++) {
    std::this_thread::yield();
    for (uptr j = 0; j < kThreadCount; j++)
      if (__sync_fetch_and_add(&callback_argument->counters[j], 0) !=
          counters_at_init[j]) {
        callback_argument->threads_stopped = false;
        return;
      }
  }
  callback_argument->threads_stopped = true;
}

TEST(StopTheWorld, SuspendThreadsAdvanced) {
  AdvancedCallbackArgument argument;

  advanced_incrementer_thread_exit_mutex.lock();
  argument.threads[0] =
      std::thread(AdvancedIncrementerThread, std::ref(argument));
  // Wait for several threads to spawn before proceeding.
  while (__sync_fetch_and_add(&argument.thread_index, 0) < kStopWorldAfter)
    std::this_thread::yield();
  StopTheWorld(&AdvancedCallback, &argument);
  EXPECT_TRUE(argument.callback_executed);
  EXPECT_TRUE(argument.threads_stopped);

  // Wait for all threads to spawn before we start terminating them.
  while (__sync_fetch_and_add(&argument.thread_index, 0) < kThreadCount)
    std::this_thread::yield();
  ASSERT_FALSE(argument.fatal_error);  // a thread could not be started
  // Signal the threads to terminate.
  advanced_incrementer_thread_exit_mutex.unlock();
  for (uptr i = 0; i < kThreadCount; i++) argument.threads[i].join();
}

static void SegvCallback(const SuspendedThreadsList &suspended_threads_list,
                         void *argument) {
  *(volatile int *)0x1234 = 0;
}

TEST(StopTheWorld, SegvInCallback) {
  // Test that tracer thread catches SIGSEGV.
  StopTheWorld(&SegvCallback, NULL);
}

}  // namespace __sanitizer

#endif  // SANITIZER_LINUX && defined(__x86_64__)
