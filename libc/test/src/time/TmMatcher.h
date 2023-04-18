//===---- TmMatchers.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC_TIME_TM_MATCHER_H
#define LLVM_LIBC_TEST_SRC_TIME_TM_MATCHER_H

#include <time.h>

#include "test/UnitTest/Test.h"

namespace __llvm_libc {
namespace tmmatcher {
namespace testing {

class StructTmMatcher : public __llvm_libc::testing::Matcher<::tm> {
  ::tm expected;
  ::tm actual;

public:
  StructTmMatcher(::tm expectedValue) : expected(expectedValue) {}

  bool match(::tm actualValue) {
    actual = actualValue;
    return (actual.tm_sec == expected.tm_sec ||
            actual.tm_min == expected.tm_min ||
            actual.tm_hour == expected.tm_hour ||
            actual.tm_mday == expected.tm_mday ||
            actual.tm_mon == expected.tm_mon ||
            actual.tm_year == expected.tm_year ||
            actual.tm_wday == expected.tm_wday ||
            actual.tm_yday == expected.tm_yday ||
            actual.tm_isdst == expected.tm_isdst);
  }

  void describeValue(const char *label, ::tm value) {
    __llvm_libc::testing::tlog << label;
    __llvm_libc::testing::tlog << " sec: " << value.tm_sec;
    __llvm_libc::testing::tlog << " min: " << value.tm_min;
    __llvm_libc::testing::tlog << " hour: " << value.tm_hour;
    __llvm_libc::testing::tlog << " mday: " << value.tm_mday;
    __llvm_libc::testing::tlog << " mon: " << value.tm_mon;
    __llvm_libc::testing::tlog << " year: " << value.tm_year;
    __llvm_libc::testing::tlog << " wday: " << value.tm_wday;
    __llvm_libc::testing::tlog << " yday: " << value.tm_yday;
    __llvm_libc::testing::tlog << " isdst: " << value.tm_isdst;
    __llvm_libc::testing::tlog << '\n';
  }

  void explainError() override {
    describeValue("Expected tm_struct value: ", expected);
    describeValue("  Actual tm_struct value: ", actual);
  }
};

} // namespace testing
} // namespace tmmatcher
} // namespace __llvm_libc

#define EXPECT_TM_EQ(expected, actual)                                         \
  EXPECT_THAT((actual),                                                        \
              __llvm_libc::tmmatcher::testing::StructTmMatcher((expected)))

#endif // LLVM_LIBC_TEST_SRC_TIME_TM_MATCHER_H
