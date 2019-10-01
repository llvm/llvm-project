//===-- sanitizer_stoptheworld_test.cc ------------------------------------===//
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

#include "sanitizer_common/sanitizer_platform.h"
#if SANITIZER_LINUX && defined(__x86_64__)

#include "sanitizer_common/sanitizer_stoptheworld.h"
#include "gtest/gtest.h"

#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_common.h"

#include <pthread.h>
#include <sched.h>

namespace __sanitizer {

static pthread_mutex_t incrementer_thread_exit_mutex;

struct CallbackArgument {
  volatile int counter;
  volatile bool threads_stopped;
  volatile bool callback_executed;
  CallbackArgument()
    : counter(0),
      threads_stopped(false),
      callback_executed(false) {}
};

void *IncrementerThread(void *argument) {
  CallbackArgument *callback_argument = (CallbackArgument *)argument;
  while (true) {
    __sync_fetch_and_add(&callback_argument->counter, 1);
    if (pthread_mutex_trylock(&incrementer_thread_exit_mutex) == 0) {
      pthread_mutex_unlock(&incrementer_thread_exit_mutex);
      return NULL;
    } else {
      sched_yield();
    }
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
    sched_yield();
    if (__sync_fetch_and_add(&callback_argument->counter, 0) !=
          counter_at_init) {
      callback_argument->threads_stopped = false;
      return;
    }
  }
  callback_argument->threads_stopped = true;
}

TEST(StopTheWorld, SuspendThreadsSimple) {
  pthread_mutex_init(&incrementer_thread_exit_mutex, NULL);
  CallbackArgument argument;
  pthread_t thread_id;
  int pthread_create_result;
  pthread_mutex_lock(&incrementer_thread_exit_mutex);
  pthread_create_result = pthread_create(&thread_id, NULL, IncrementerThread,
                                         &argument);
  ASSERT_EQ(0, pthread_create_result);
  StopTheWorld(&Callback, &argument);
  pthread_mutex_unlock(&incrementer_thread_exit_mutex);
  EXPECT_TRUE(argument.callback_executed);
  EXPECT_TRUE(argument.threads_stopped);
  // argument is on stack, so we have to wait for the incrementer thread to
  // terminate before we can return from this function.
  ASSERT_EQ(0, pthread_join(thread_id, NULL));
  pthread_mutex_destroy(&incrementer_thread_exit_mutex);
}

// A more comprehensive test where we spawn a bunch of threads while executing
// StopTheWorld in parallel.
static const uptr kThreadCount = 50;
static const uptr kStopWorldAfter = 10; // let this many threads spawn first

static pthread_mutex_t advanced_incrementer_thread_exit_mutex;

struct AdvancedCallbackArgument {
  volatile uptr thread_index;
  volatile int counters[kThreadCount];
  pthread_t thread_ids[kThreadCount];
  volatile bool threads_stopped;
  volatile bool callback_executed;
  volatile bool fatal_error;
  AdvancedCallbackArgument()
    : thread_index(0),
      threads_stopped(false),
      callback_executed(false),
      fatal_error(false) {}
};

void *AdvancedIncrementerThread(void *argument) {
  AdvancedCallbackArgument *callback_argument =
      (AdvancedCallbackArgument *)argument;
  uptr this_thread_index = __sync_fetch_and_add(
      &callback_argument->thread_index, 1);
  // Spawn the next thread.
  int pthread_create_result;
  if (this_thread_index + 1 < kThreadCount) {
    pthread_create_result =
        pthread_create(&callback_argument->thread_ids[this_thread_index + 1],
                       NULL, AdvancedIncrementerThread, argument);
    // Cannot use ASSERT_EQ in non-void-returning functions. If there's a
    // problem, defer failing to the main thread.
    if (pthread_create_result != 0) {
      callback_argument->fatal_error = true;
      __sync_fetch_and_add(&callback_argument->thread_index,
                           kThreadCount - callback_argument->thread_index);
    }
  }
  // Do the actual work.
  while (true) {
    __sync_fetch_and_add(&callback_argument->counters[this_thread_index], 1);
    if (pthread_mutex_trylock(&advanced_incrementer_thread_exit_mutex) == 0) {
      pthread_mutex_unlock(&advanced_incrementer_thread_exit_mutex);
      return NULL;
    } else {
      sched_yield();
    }
  }
}

void AdvancedCallback(const SuspendedThreadsList &suspended_threads_list,
                             void *argument) {
  AdvancedCallbackArgument *callback_argument =
      (AdvancedCallbackArgument *)argument;
  callback_argument->callback_executed = true;

  int counters_at_init[kThreadCount];
  for (uptr j = 0; j < kThreadCount; j++)
    counters_at_init[j] = __sync_fetch_and_add(&callback_argument->counters[j],
                                               0);
  for (uptr i = 0; i < 10; i++) {
    sched_yield();
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
  pthread_mutex_init(&advanced_incrementer_thread_exit_mutex, NULL);
  AdvancedCallbackArgument argument;

  pthread_mutex_lock(&advanced_incrementer_thread_exit_mutex);
  int pthread_create_result;
  pthread_create_result = pthread_create(&argument.thread_ids[0], NULL,
                                         AdvancedIncrementerThread,
                                         &argument);
  ASSERT_EQ(0, pthread_create_result);
  // Wait for several threads to spawn before proceeding.
  while (__sync_fetch_and_add(&argument.thread_index, 0) < kStopWorldAfter)
    sched_yield();
  StopTheWorld(&AdvancedCallback, &argument);
  EXPECT_TRUE(argument.callback_executed);
  EXPECT_TRUE(argument.threads_stopped);

  // Wait for all threads to spawn before we start terminating them.
  while (__sync_fetch_and_add(&argument.thread_index, 0) < kThreadCount)
    sched_yield();
  ASSERT_FALSE(argument.fatal_error); // a pthread_create has failed
  // Signal the threads to terminate.
  pthread_mutex_unlock(&advanced_incrementer_thread_exit_mutex);
  for (uptr i = 0; i < kThreadCount; i++)
    ASSERT_EQ(0, pthread_join(argument.thread_ids[i], NULL));
  pthread_mutex_destroy(&advanced_incrementer_thread_exit_mutex);
}

static void SegvCallback(const SuspendedThreadsList &suspended_threads_list,
                         void *argument) {
  *(volatile int*)0x1234 = 0;
}

TEST(StopTheWorld, SegvInCallback) {
  // Test that tracer thread catches SIGSEGV.
  StopTheWorld(&SegvCallback, NULL);
}

}  // namespace __sanitizer

#endif  // SANITIZER_LINUX && defined(__x86_64__)
