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
#include "test/UnitTest/Test.h"

#include <errno.h>

namespace LIBC_NAMESPACE {
namespace testing {

class FormatSectionMatcher : public Matcher<printf_core::FormatSection> {
  printf_core::FormatSection expected;
  printf_core::FormatSection actual;

public:
  FormatSectionMatcher(printf_core::FormatSection expectedValue)
      : expected(expectedValue) {}

  bool match(printf_core::FormatSection actualValue);

  void explainError() override;
};

} // namespace testing
} // namespace LIBC_NAMESPACE

#define EXPECT_PFORMAT_EQ(expected, actual)                                    \
  EXPECT_THAT(actual, LIBC_NAMESPACE::testing::FormatSectionMatcher(expected))

#define ASSERT_PFORMAT_EQ(expected, actual)                                    \
  ASSERT_THAT(actual, LIBC_NAMESPACE::testing::FormatSectionMatcher(expected))

#endif // LLVM_LIBC_UTILS_UNITTEST_PRINTF_MATCHER_H
