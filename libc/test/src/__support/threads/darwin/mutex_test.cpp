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
  ;
  ASSERT_EQ(mutex.lock(), LIBC_NAMESPACE::MutexError::NONE);
  ASSERT_EQ(mutex.unlock(), LIBC_NAMESPACE::MutexError::NONE);
  ASSERT_EQ(mutex.try_lock(), LIBC_NAMESPACE::MutexError::NONE);
  ASSERT_EQ(mutex.try_lock(), LIBC_NAMESPACE::MutexError::BUSY);
  ASSERT_EQ(mutex.unlock(), LIBC_NAMESPACE::MutexError::NONE);
  ASSERT_EQ(mutex.unlock(), LIBC_NAMESPACE::MutexError::UNLOCK_WITHOUT_LOCK);
}

// TODO(bojle): add other tests a la linux
