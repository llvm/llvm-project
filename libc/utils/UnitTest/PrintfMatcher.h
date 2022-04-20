//===-- PrintfMatcher.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_UNITTEST_PRINTF_MATCHER_H
#define LLVM_LIBC_UTILS_UNITTEST_PRINTF_MATCHER_H

#include "src/stdio/printf_core/core_structs.h"
#include "utils/UnitTest/Test.h"

#include <errno.h>

namespace __llvm_libc {
namespace printf_core {
namespace testing {

class FormatSectionMatcher
    : public __llvm_libc::testing::Matcher<FormatSection> {
  FormatSection expected;
  FormatSection actual;

public:
  FormatSectionMatcher(FormatSection expectedValue) : expected(expectedValue) {}

  bool match(FormatSection actualValue);

  void explainError(testutils::StreamWrapper &stream) override;
};

} // namespace testing
} // namespace printf_core
} // namespace __llvm_libc

#define EXPECT_FORMAT_EQ(expected, actual)                                     \
  EXPECT_THAT(actual, __llvm_libc::printf_core::testing::FormatSectionMatcher( \
                          expected))

#define ASSERT_FORMAT_EQ(expected, actual)                                     \
  ASSERT_THAT(actual, __llvm_libc::printf_core::testing::FormatSectionMatcher( \
                          expected))

#endif // LLVM_LIBC_UTILS_UNITTEST_PRINTF_MATCHER_H
