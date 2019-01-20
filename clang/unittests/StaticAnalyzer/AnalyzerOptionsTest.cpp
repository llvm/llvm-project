//===- unittest/StaticAnalyzer/AnalyzerOptionsTest.cpp - SA Options test --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/AnalyzerOptions.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "gtest/gtest.h"

namespace clang {
namespace ento {

TEST(StaticAnalyzerOptions, getRegisteredCheckers) {
  auto IsDebugChecker = [](StringRef CheckerName) {
    return CheckerName.startswith("debug");
  };
  auto IsAlphaChecker = [](StringRef CheckerName) {
    return CheckerName.startswith("alpha");
  };
  const auto &AllCheckers =
      AnalyzerOptions::getRegisteredCheckers(/*IncludeExperimental=*/true);
  EXPECT_FALSE(llvm::any_of(AllCheckers, IsDebugChecker));
  EXPECT_TRUE(llvm::any_of(AllCheckers, IsAlphaChecker));

  const auto &StableCheckers = AnalyzerOptions::getRegisteredCheckers();
  EXPECT_FALSE(llvm::any_of(StableCheckers, IsDebugChecker));
  EXPECT_FALSE(llvm::any_of(StableCheckers, IsAlphaChecker));
}

TEST(StaticAnalyzerOptions, SearchInParentPackageTests) {
  AnalyzerOptions Opts;
  Opts.Config["Outer.Inner.CheckerOne:Option"] = "true";
  Opts.Config["Outer.Inner:Option"] = "false";
  Opts.Config["Outer.Inner:Option2"] = "true";
  Opts.Config["Outer:Option2"] = "false";

  struct CheckerOneMock : CheckerBase {
    StringRef getTagDescription() const override {
      return "Outer.Inner.CheckerOne";
    }
  };
  struct CheckerTwoMock : CheckerBase {
    StringRef getTagDescription() const override {
      return "Outer.Inner.CheckerTwo";
    }
  };

  // Checker one has Option specified as true. It should read true regardless of
  // search mode.
  CheckerOneMock CheckerOne;
  EXPECT_TRUE(Opts.getCheckerBooleanOption("Option", false, &CheckerOne));
  // The package option is overridden with a checker option.
  EXPECT_TRUE(Opts.getCheckerBooleanOption("Option", false, &CheckerOne,
                                           true));
  // The Outer package option is overridden by the Inner package option. No
  // package option is specified.
  EXPECT_TRUE(Opts.getCheckerBooleanOption("Option2", false, &CheckerOne,
                                           true));
  // No package option is specified and search in packages is turned off. The
  // default value should be returned.
  EXPECT_FALSE(Opts.getCheckerBooleanOption("Option2", false, &CheckerOne));
  EXPECT_TRUE(Opts.getCheckerBooleanOption("Option2", true, &CheckerOne));

  // Checker true has no option specified. It should get the default value when
  // search in parents turned off and false when search in parents turned on.
  CheckerTwoMock CheckerTwo;
  EXPECT_FALSE(Opts.getCheckerBooleanOption("Option", false, &CheckerTwo));
  EXPECT_TRUE(Opts.getCheckerBooleanOption("Option", true, &CheckerTwo));
  EXPECT_FALSE(Opts.getCheckerBooleanOption("Option", true, &CheckerTwo, true));
}

TEST(StaticAnalyzerOptions, StringOptions) {
  AnalyzerOptions Opts;
  Opts.Config["Outer.Inner.CheckerOne:Option"] = "StringValue";

  struct CheckerOneMock : CheckerBase {
    StringRef getTagDescription() const override {
      return "Outer.Inner.CheckerOne";
    }
  };

  CheckerOneMock CheckerOne;
  EXPECT_TRUE("StringValue" ==
            Opts.getCheckerStringOption("Option", "DefaultValue", &CheckerOne));
  EXPECT_TRUE("DefaultValue" ==
           Opts.getCheckerStringOption("Option2", "DefaultValue", &CheckerOne));
}
} // end namespace ento
} // end namespace clang
