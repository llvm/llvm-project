//===-- flang/unittests/Common/FastIntSetTest.cpp ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "flang/Common/enum-class.h"
#include "flang/Common/template.h"

using namespace Fortran::common;
using namespace std;

ENUM_CLASS(TestEnum, One, Two, Three)
ENUM_CLASS_EXTRA(TestEnum)

TEST(EnumClassTest, EnumToString) {
  ASSERT_EQ(EnumToString(TestEnum::One), "One");
  ASSERT_EQ(EnumToString(TestEnum::Two), "Two");
  ASSERT_EQ(EnumToString(TestEnum::Three), "Three");
}

TEST(EnumClassTest, EnumToStringData) {
  ASSERT_STREQ(EnumToString(TestEnum::One).data(), "One, Two, Three");
}

TEST(EnumClassTest, StringToEnum) {
  ASSERT_EQ(StringToTestEnum("One"), std::optional{TestEnum::One});
  ASSERT_EQ(StringToTestEnum("Two"), std::optional{TestEnum::Two});
  ASSERT_EQ(StringToTestEnum("Three"), std::optional{TestEnum::Three});
  ASSERT_EQ(StringToTestEnum("Four"), std::nullopt);
  ASSERT_EQ(StringToTestEnum(""), std::nullopt);
  ASSERT_EQ(StringToTestEnum("One, Two, Three"), std::nullopt);
}

ENUM_CLASS(TestEnumExtra, TwentyOne, FortyTwo, SevenSevenSeven)
ENUM_CLASS_EXTRA(TestEnumExtra)

TEST(EnumClassTest, FindNameNormal) {
  auto p1 = [](auto s) { return s == "TwentyOne"; };
  ASSERT_EQ(FindTestEnumExtra(p1), std::optional{TestEnumExtra::TwentyOne});
}
