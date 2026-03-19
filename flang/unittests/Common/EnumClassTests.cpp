//===-- flang/unittests/Common/FastIntSetTest.cpp ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "flang/Common/enum-class.h"

namespace Fortran::common {

ENUM_CLASS(TestEnum, One, Two, Three)

TEST(EnumClassTest, EnumToString) {
  ASSERT_EQ(EnumToString(TestEnum::One), "One");
  ASSERT_EQ(EnumToString(TestEnum::Two), "Two");
  ASSERT_EQ(EnumToString(TestEnum::Three), "Three");
}

TEST(EnumClassTest, EnumClassForEach) {
  std::string result;
  bool first{true};
  ForEachTestEnum([&](auto e) {
    if (!first) {
      result += ", ";
    }
    result += EnumToString(e);
    first = false;
  });
  ASSERT_EQ(result, "One, Two, Three");
}
} // namespace Fortran::common
