//===-- Unittests for Atomic ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/atomic.h"
#include "utils/UnitTest/Test.h"

// Tests in this file do not test atomicity as it would require using
// threads, at which point it becomes a chicken and egg problem.

TEST(LlvmLibcAtomicTest, LoadStore) {
  __llvm_libc::cpp::Atomic<int> aint(123);
  ASSERT_EQ(aint.load(), 123);

  aint.store(100);
  ASSERT_EQ(aint.load(), 100);

  aint = 1234; // Equivalent of store
  ASSERT_EQ(aint.load(), 1234);
}

TEST(LlvmLibcAtomicTest, CompareExchangeStrong) {
  int desired = 123;
  __llvm_libc::cpp::Atomic<int> aint(desired);
  ASSERT_TRUE(aint.compare_exchange_strong(desired, 100));
  ASSERT_EQ(aint.load(), 100);

  ASSERT_FALSE(aint.compare_exchange_strong(desired, 100));
  ASSERT_EQ(aint.load(), 100);
}
