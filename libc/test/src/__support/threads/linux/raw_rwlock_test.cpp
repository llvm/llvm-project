//===-- Unittests for RawRwLock -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/time_macros.h"
#include "src/__support/threads/raw_rwlock.h"
#include "src/__support/time/clock_gettime.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcSupportThreadsRawRwLockTest, SmokeTest) {
  LIBC_NAMESPACE::RawRwLock rwlock;
  using LockResult = LIBC_NAMESPACE::RawRwLock::LockResult;

  ASSERT_EQ(rwlock.read_lock(), LockResult::Success);
  ASSERT_EQ(rwlock.try_read_lock(), LockResult::Success);
  ASSERT_EQ(rwlock.try_write_lock(), LockResult::Busy);
  ASSERT_EQ(rwlock.unlock(), LockResult::Success);
  ASSERT_EQ(rwlock.unlock(), LockResult::Success);

  ASSERT_EQ(rwlock.write_lock(), LockResult::Success);
  ASSERT_EQ(rwlock.try_read_lock(), LockResult::Busy);
  ASSERT_EQ(rwlock.try_write_lock(), LockResult::Busy);
  ASSERT_EQ(rwlock.unlock(), LockResult::Success);
  ASSERT_EQ(rwlock.unlock(), LockResult::PermissionDenied);
}

TEST(LlvmLibcSupportThreadsRawRwLockTest, TimeoutWithoutDeadlockDetection) {
  LIBC_NAMESPACE::RawRwLock rwlock;
  using LockResult = LIBC_NAMESPACE::RawRwLock::LockResult;

  ASSERT_EQ(rwlock.write_lock(), LockResult::Success);

  timespec ts;
  LIBC_NAMESPACE::internal::clock_gettime(CLOCK_MONOTONIC, &ts);
  ts.tv_sec += 1;
  auto timeout = LIBC_NAMESPACE::internal::AbsTimeout::from_timespec(
      ts, /*is_realtime=*/false);
  ASSERT_TRUE(timeout.has_value());

  ASSERT_EQ(rwlock.write_lock(*timeout), LockResult::TimedOut);
  ASSERT_EQ(rwlock.read_lock(*timeout), LockResult::TimedOut);
  ASSERT_EQ(rwlock.unlock(), LockResult::Success);
}
