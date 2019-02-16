//===- unittest/StaticAnalyzer/AnalyzerOptionsTest.cpp - SA Options test --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
  EXPECT_TRUE(Opts.getBooleanOption("Option", false, &CheckerOne));
  // The package option is overridden with a checker option.
  EXPECT_TRUE(Opts.getBooleanOption("Option", false, &CheckerOne, true));
  // The Outer package option is overridden by the Inner package option. No
  // package option is specified.
  EXPECT_TRUE(Opts.getBooleanOption("Option2", false, &CheckerOne, true));
  // No package option is specified and search in packages is turned off. The
  // default value should be returned.
  EXPECT_FALSE(Opts.getBooleanOption("Option2", false, &CheckerOne));
  EXPECT_TRUE(Opts.getBooleanOption("Option2", true, &CheckerOne));

  // Checker true has no option specified. It should get the default value when
  // search in parents turned off and false when search in parents turned on.
  CheckerTwoMock CheckerTwo;
  EXPECT_FALSE(Opts.getBooleanOption("Option", false, &CheckerTwo));
  EXPECT_TRUE(Opts.getBooleanOption("Option", true, &CheckerTwo));
  EXPECT_FALSE(Opts.getBooleanOption("Option", true, &CheckerTwo, true));
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
              Opts.getOptionAsString("Option", "DefaultValue", &CheckerOne));
  EXPECT_TRUE("DefaultValue" ==
              Opts.getOptionAsString("Option2", "DefaultValue", &CheckerOne));
}
} // end namespace ento
} // end namespace clang
