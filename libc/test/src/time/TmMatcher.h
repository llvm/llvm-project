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

#include "utils/UnitTest/Test.h"

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

  void describeValue(const char *label, ::tm value,
                     __llvm_libc::testutils::StreamWrapper &stream) {
    stream << label;
    stream << " sec: " << value.tm_sec;
    stream << " min: " << value.tm_min;
    stream << " hour: " << value.tm_hour;
    stream << " mday: " << value.tm_mday;
    stream << " mon: " << value.tm_mon;
    stream << " year: " << value.tm_year;
    stream << " wday: " << value.tm_wday;
    stream << " yday: " << value.tm_yday;
    stream << " isdst: " << value.tm_isdst;
    stream << '\n';
  }

  void explainError(__llvm_libc::testutils::StreamWrapper &stream) override {
    describeValue("Expected tm_struct value: ", expected, stream);
    describeValue("  Actual tm_struct value: ", actual, stream);
  }
};

} // namespace testing
} // namespace tmmatcher
} // namespace __llvm_libc

#define EXPECT_TM_EQ(expected, actual)                                         \
  EXPECT_THAT((actual),                                                        \
              __llvm_libc::tmmatcher::testing::StructTmMatcher((expected)))

#endif // LLVM_LIBC_TEST_SRC_TIME_TM_MATCHER_H
