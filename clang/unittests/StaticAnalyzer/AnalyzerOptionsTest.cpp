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
    return CheckerName.starts_with("debug");
  };
  auto IsAlphaChecker = [](StringRef CheckerName) {
    return CheckerName.starts_with("alpha");
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

  // CheckerTwo one has Option specified as true. It should read true regardless
  // of search mode.
  CheckerOneMock CheckerOne;
  EXPECT_TRUE(Opts.getCheckerBooleanOption(&CheckerOne, "Option"));
  // The package option is overridden with a checker option.
  EXPECT_TRUE(Opts.getCheckerBooleanOption(&CheckerOne, "Option", true));
  // The Outer package option is overridden by the Inner package option. No
  // package option is specified.
  EXPECT_TRUE(Opts.getCheckerBooleanOption(&CheckerOne, "Option2", true));
  // No package option is specified and search in packages is turned off. We
  // should assert here, but we can't test that.
  //Opts.getCheckerBooleanOption(&CheckerOne, "Option2");
  //Opts.getCheckerBooleanOption(&CheckerOne, "Option2");

  // Checker true has no option specified. It should get false when search in
  // parents turned on.
  CheckerTwoMock CheckerTwo;
  EXPECT_FALSE(Opts.getCheckerBooleanOption(&CheckerTwo, "Option", true));
  // In any other case, we should assert, that we cannot test unfortunately.
  //Opts.getCheckerBooleanOption(&CheckerTwo, "Option");
  //Opts.getCheckerBooleanOption(&CheckerTwo, "Option");
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
            Opts.getCheckerStringOption(&CheckerOne, "Option"));
}

TEST(StaticAnalyzerOptions, SubCheckerOptions) {
  AnalyzerOptions Opts;
  Opts.Config["Outer.Inner.CheckerOne:Option"] = "StringValue";
  EXPECT_TRUE("StringValue" == Opts.getCheckerStringOption(
        "Outer.Inner.CheckerOne", "Option"));
}

} // end namespace ento
} // end namespace clang
