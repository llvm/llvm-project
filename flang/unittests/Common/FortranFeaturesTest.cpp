//===-- flang/unittests/Common/FastIntSetTest.cpp ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "flang/Common/enum-class.h"
#include "flang/Support/Fortran-features.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include <optional>

namespace Fortran::common::FortranFeaturesHelpers {

optional<std::pair<bool, UsageWarning>> parseCLIUsageWarning(
    llvm::StringRef input);
TEST(EnumClassTest, ParseCLIUsageWarning) {
  EXPECT_EQ((parseCLIUsageWarning("no-twenty-one")), std::nullopt);
  EXPECT_EQ((parseCLIUsageWarning("twenty-one")), std::nullopt);
  EXPECT_EQ((parseCLIUsageWarning("no-seven-seven-seven")), std::nullopt);
  EXPECT_EQ((parseCLIUsageWarning("seven-seven-seven")), std::nullopt);
  EXPECT_EQ((parseCLIUsageWarning("no")), std::nullopt);
  EXPECT_EQ((parseCLIUsageWarning("")), std::nullopt);
  EXPECT_EQ((parseCLIUsageWarning("no-")), std::nullopt);
  EXPECT_EQ(parseCLIUsageWarning("Portability"), std::nullopt);
  auto expect{std::pair{false, UsageWarning::Portability}};
  ASSERT_EQ(parseCLIUsageWarning("no-portability"), expect);
  expect.first = true;
  ASSERT_EQ((parseCLIUsageWarning("portability")), expect);
  expect =
      std::pair{false, Fortran::common::UsageWarning::PointerToUndefinable};
  ASSERT_EQ((parseCLIUsageWarning("no-pointer-to-undefinable")), expect);
  expect.first = true;
  ASSERT_EQ((parseCLIUsageWarning("pointer-to-undefinable")), expect);
  EXPECT_EQ(parseCLIUsageWarning("PointerToUndefinable"), std::nullopt);
  EXPECT_EQ(parseCLIUsageWarning("NoPointerToUndefinable"), std::nullopt);
  EXPECT_EQ(parseCLIUsageWarning("pointertoundefinable"), std::nullopt);
  EXPECT_EQ(parseCLIUsageWarning("nopointertoundefinable"), std::nullopt);
}

} // namespace Fortran::common::FortranFeaturesHelpers
