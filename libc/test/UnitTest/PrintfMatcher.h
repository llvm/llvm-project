//===-- PrintfMatcher.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_UNITTEST_PRINTF_MATCHER_H
#define LLVM_LIBC_UTILS_UNITTEST_PRINTF_MATCHER_H

#include "src/__support/macros/config.h"
#include "src/__support/printf_core/core_structs.h"
#include "test/UnitTest/Test.h"

namespace LIBC_NAMESPACE_DECL {
namespace testing {

void display(MatchResultListener &listener,
             const printf_core::FormatSection<char> &format_section);
void display(MatchResultListener &listener,
             const printf_core::FormatSection<wchar_t> &format_section);

template <typename CharT>
class FormatSectionMatcher : public Matcher<printf_core::FormatSection<CharT>> {
  printf_core::FormatSection<CharT> expected;

public:
  FormatSectionMatcher(printf_core::FormatSection<CharT> expectedValue)
      : expected(expectedValue) {}

  bool MatchAndExplain(const printf_core::FormatSection<CharT> &actual,
                       MatchResultListener *listener) const {
    if (expected == actual)
      return true;
    if (!listener->IsInterested())
      return false;
    *listener << "expected format section: ";
    display(*listener, expected);
    *listener << "\nactual format section  : ";
    display(*listener, actual);
    *listener << '\n';
    return false;
  }
};

template <typename CharT>
FormatSectionMatcher<CharT>
MakeFormatSectionMatcher(printf_core::FormatSection<CharT> expected) {
  return FormatSectionMatcher<CharT>(expected);
}

} // namespace testing
} // namespace LIBC_NAMESPACE_DECL

#define EXPECT_PFORMAT_EQ(expected, actual)                                    \
  EXPECT_THAT(actual,                                                          \
              LIBC_NAMESPACE::testing::MakeFormatSectionMatcher(expected))

#define ASSERT_PFORMAT_EQ(expected, actual)                                    \
  ASSERT_THAT(actual,                                                          \
              LIBC_NAMESPACE::testing::MakeFormatSectionMatcher(expected))

#endif // LLVM_LIBC_UTILS_UNITTEST_PRINTF_MATCHER_H
