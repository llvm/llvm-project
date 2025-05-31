//===-- flang/unittests/Common/FastIntSetTest.cpp ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "flang/Support/Fortran-features.h"
#include <optional>

namespace Fortran::common::FortranFeaturesHelpers {

optional<std::pair<bool, UsageWarning>> parseCLIUsageWarning(
    llvm::StringRef input, bool insensitive);
TEST(EnumClassTest, ParseCLIUsageWarning) {
  EXPECT_EQ((parseCLIUsageWarning("no-twenty-one", false)), std::nullopt);
  EXPECT_EQ((parseCLIUsageWarning("twenty-one", false)), std::nullopt);
  EXPECT_EQ(
      (parseCLIUsageWarning("no-seven-seven-seven", false)), std::nullopt);
  EXPECT_EQ((parseCLIUsageWarning("seven-seven-seven", false)), std::nullopt);
  EXPECT_EQ((parseCLIUsageWarning("no", false)), std::nullopt);
  EXPECT_EQ((parseCLIUsageWarning("", false)), std::nullopt);
  EXPECT_EQ((parseCLIUsageWarning("no-", false)), std::nullopt);

  EXPECT_EQ(parseCLIUsageWarning("Portability", false), std::nullopt);
  EXPECT_EQ((parseCLIUsageWarning("no-portability", false)),
      (std::optional{std::pair{false, UsageWarning::Portability}}));
  EXPECT_EQ((parseCLIUsageWarning("portability", false)),
      (std::optional{std::pair{true, UsageWarning::Portability}}));
  EXPECT_EQ((parseCLIUsageWarning("no-pointer-to-undefinable", false)),
      (std::optional{std::pair{false, UsageWarning::PointerToUndefinable}}));
  EXPECT_EQ((parseCLIUsageWarning("pointer-to-undefinable", false)),
      (std::optional{std::pair{true, UsageWarning::PointerToUndefinable}}));
  EXPECT_EQ(parseCLIUsageWarning("PointerToUndefinable", false), std::nullopt);
  EXPECT_EQ(
      parseCLIUsageWarning("NoPointerToUndefinable", false), std::nullopt);
  EXPECT_EQ(parseCLIUsageWarning("pointertoundefinable", false), std::nullopt);
  EXPECT_EQ(
      parseCLIUsageWarning("nopointertoundefinable", false), std::nullopt);

  EXPECT_EQ((parseCLIUsageWarning("no-twenty-one", false)), std::nullopt);
  EXPECT_EQ((parseCLIUsageWarning("twenty-one", true)), std::nullopt);
  EXPECT_EQ((parseCLIUsageWarning("no-seven-seven-seven", true)), std::nullopt);
  EXPECT_EQ((parseCLIUsageWarning("seven-seven-seven", true)), std::nullopt);
  EXPECT_EQ((parseCLIUsageWarning("no", true)), std::nullopt);
  EXPECT_EQ((parseCLIUsageWarning("", true)), std::nullopt);
  EXPECT_EQ((parseCLIUsageWarning("no-", true)), std::nullopt);

  EXPECT_EQ(parseCLIUsageWarning("Portability", true),
      (std::optional{std::pair{true, UsageWarning::Portability}}));
  EXPECT_EQ(parseCLIUsageWarning("no-portability", true),
      (std::optional{std::pair{false, UsageWarning::Portability}}));

  EXPECT_EQ((parseCLIUsageWarning("portability", true)),
      (std::optional{std::pair{true, UsageWarning::Portability}}));
  EXPECT_EQ((parseCLIUsageWarning("no-pointer-to-undefinable", true)),
      (std::optional{std::pair{false, UsageWarning::PointerToUndefinable}}));
  EXPECT_EQ((parseCLIUsageWarning("pointer-to-undefinable", true)),
      (std::optional{std::pair{true, UsageWarning::PointerToUndefinable}}));
  EXPECT_EQ(parseCLIUsageWarning("PointerToUndefinable", true),
      (std::optional{std::pair{true, UsageWarning::PointerToUndefinable}}));
  EXPECT_EQ(parseCLIUsageWarning("NoPointerToUndefinable", true),
      (std::optional{std::pair{false, UsageWarning::PointerToUndefinable}}));
  EXPECT_EQ(parseCLIUsageWarning("pointertoundefinable", true),
      (std::optional{std::pair{true, UsageWarning::PointerToUndefinable}}));
  EXPECT_EQ(parseCLIUsageWarning("nopointertoundefinable", true),
      (std::optional{std::pair{false, UsageWarning::PointerToUndefinable}}));
}

} // namespace Fortran::common::FortranFeaturesHelpers
