//===-- ScanfMatcher.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_UNITTEST_SCANF_MATCHER_H
#define LLVM_LIBC_UTILS_UNITTEST_SCANF_MATCHER_H

#include "src/stdio/scanf_core/core_structs.h"
#include "test/UnitTest/Test.h"

#include <errno.h>

namespace __llvm_libc {
namespace testing {

class FormatSectionMatcher : public Matcher<scanf_core::FormatSection> {
  scanf_core::FormatSection expected;
  scanf_core::FormatSection actual;

public:
  FormatSectionMatcher(scanf_core::FormatSection expectedValue)
      : expected(expectedValue) {}

  bool match(scanf_core::FormatSection actualValue);

  void explainError() override;
};

} // namespace testing
} // namespace __llvm_libc

#define EXPECT_SFORMAT_EQ(expected, actual)                                    \
  EXPECT_THAT(actual, __llvm_libc::testing::FormatSectionMatcher(expected))

#define ASSERT_SFORMAT_EQ(expected, actual)                                    \
  ASSERT_THAT(actual, __llvm_libc::testing::FormatSectionMatcher(expected))

#endif // LLVM_LIBC_UTILS_UNITTEST_SCANF_MATCHER_H
