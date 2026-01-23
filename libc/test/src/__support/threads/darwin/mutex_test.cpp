//===-- Unittests for Darwin's Mutex ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/threads/mutex.h"
#include "src/__support/threads/mutex_common.h"
#include "src/__support/threads/raw_mutex.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcSupportThreadsMutexTest, SmokeTest) {
  LIBC_NAMESPACE::Mutex mutex(0, 0, 0, 0);
  ASSERT_EQ(mutex.lock(), LIBC_NAMESPACE::MutexError::NONE);
  ASSERT_EQ(mutex.unlock(), LIBC_NAMESPACE::MutexError::NONE);
  ASSERT_EQ(mutex.try_lock(), LIBC_NAMESPACE::MutexError::NONE);
  ASSERT_EQ(mutex.try_lock(), LIBC_NAMESPACE::MutexError::BUSY);
  ASSERT_EQ(mutex.unlock(), LIBC_NAMESPACE::MutexError::NONE);
  ASSERT_EQ(mutex.unlock(), LIBC_NAMESPACE::MutexError::UNLOCK_WITHOUT_LOCK);
}

TEST(LlvmLibcSupportThreadsRawMutexTest, Timeout) {
  LIBC_NAMESPACE::RawMutex mutex;
  ASSERT_TRUE(mutex.lock());
  timespec ts;
  LIBC_NAMESPACE::internal::clock_gettime(CLOCK_MONOTONIC, &ts);
  ts.tv_sec += 1;
  // Timeout will be respected when deadlock happens.
  auto timeout = LIBC_NAMESPACE::internal::AbsTimeout::from_timespec(ts, false);
  ASSERT_TRUE(timeout.has_value());
  // The following will timeout
  ASSERT_FALSE(mutex.lock(*timeout));
  ASSERT_TRUE(mutex.unlock());
  // Test that the mutex works after the timeout.
  ASSERT_TRUE(mutex.lock());
  ASSERT_TRUE(mutex.unlock());
  // If a lock can be acquired directly, expired timeout will not count.
  // Notice that the timeout is already reached during preivous deadlock.
  ASSERT_TRUE(mutex.lock(*timeout));
  ASSERT_TRUE(mutex.unlock());
}

// TODO(bojle): merge threads test for darwin and linux into one after
// adding support for shared locks in darwin
