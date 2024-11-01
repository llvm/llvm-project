//===-- MemoryMatcher.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_UNITTEST_MEMORY_MATCHER_H
#define LLVM_LIBC_UTILS_UNITTEST_MEMORY_MATCHER_H

#include "src/__support/CPP/span.h"

#include "test/UnitTest/Test.h"

namespace __llvm_libc {
namespace memory {
namespace testing {

using MemoryView = __llvm_libc::cpp::span<const char>;

class MemoryMatcher : public __llvm_libc::testing::Matcher<MemoryView> {
  MemoryView expected;
  MemoryView actual;
  bool mismatch_size = false;
  size_t mismatch_index = -1;

public:
  MemoryMatcher(MemoryView expectedValue) : expected(expectedValue) {}

  bool match(MemoryView actualValue);

  void explainError(testutils::StreamWrapper &stream) override;
};

} // namespace testing
} // namespace memory
} // namespace __llvm_libc

#define EXPECT_MEM_EQ(expected, actual)                                        \
  EXPECT_THAT(actual, __llvm_libc::memory::testing::MemoryMatcher(expected))
#define ASSERT_MEM_EQ(expected, actual)                                        \
  ASSERT_THAT(actual, __llvm_libc::memory::testing::MemoryMatcher(expected))

#endif // LLVM_LIBC_UTILS_UNITTEST_MEMORY_MATCHER_H
