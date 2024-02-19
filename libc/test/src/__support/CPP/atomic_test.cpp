//===-- Unittests for Atomic ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/atomic.h"
#include "test/UnitTest/Test.h"

// Tests in this file do not test atomicity as it would require using
// threads, at which point it becomes a chicken and egg problem.

TEST(LlvmLibcAtomicTest, LoadStore) {
  LIBC_NAMESPACE::cpp::Atomic<int> aint(123);
  ASSERT_EQ(aint.load(LIBC_NAMESPACE::cpp::MemoryOrder::RELAXED), 123);

  aint.store(100, LIBC_NAMESPACE::cpp::MemoryOrder::RELAXED);
  ASSERT_EQ(aint.load(LIBC_NAMESPACE::cpp::MemoryOrder::RELAXED), 100);

  aint = 1234; // Equivalent of store
  ASSERT_EQ(aint.load(LIBC_NAMESPACE::cpp::MemoryOrder::RELAXED), 1234);
}

TEST(LlvmLibcAtomicTest, CompareExchangeStrong) {
  int desired = 123;
  LIBC_NAMESPACE::cpp::Atomic<int> aint(desired);
  ASSERT_TRUE(aint.compare_exchange_strong(desired, 100));
  ASSERT_EQ(aint.load(LIBC_NAMESPACE::cpp::MemoryOrder::RELAXED), 100);

  ASSERT_FALSE(aint.compare_exchange_strong(desired, 100));
  ASSERT_EQ(aint.load(LIBC_NAMESPACE::cpp::MemoryOrder::RELAXED), 100);
}
