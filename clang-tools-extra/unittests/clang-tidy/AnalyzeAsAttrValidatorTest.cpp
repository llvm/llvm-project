//===--- AnalyzeAsAttrValidatorTest.cpp - clang-tidy ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AnalyzeAsAttrValidator.h"
#include "gtest/gtest.h"
#include <set>
#include <string>

namespace clang::tidy {
namespace {

// A table with two supported classes, each used by a different check, so the
// interaction between them can be tested before a second class is supported
// for real.
const AnalyzeAsTarget TwoTargets[] = {
    {"std::optional", "<optional>", "optional-check"},
    {"std::vector", "<vector>", "vector-check"},
};

// Builds an "is this check enabled" predicate from a set of check names.
struct Enabled {
  std::set<std::string> Checks;
  bool operator()(llvm::StringRef Check) const {
    return Checks.count(Check.str()) != 0;
  }
};

AnalyzeAsDecision classify(llvm::StringRef Target, const Enabled &On) {
  return classifyAnalyzeAsTarget(Target, TwoTargets, On);
}

// The case that motivated this test: a supported class whose check is
// disabled stays silent, even though another analyze_as check is enabled. It
// must not be mistaken for an unsupported ("misspelled") class.
TEST(AnalyzeAsClassify, SupportedClassWithItsCheckDisabledIsSilent) {
  Enabled OnlyOptional{{"optional-check"}};
  EXPECT_EQ(classify("std::vector", OnlyOptional).Action,
            AnalyzeAsDecision::Silent);
}

TEST(AnalyzeAsClassify, SupportedClassWithItsCheckEnabledIsValidated) {
  Enabled OnlyOptional{{"optional-check"}};
  AnalyzeAsDecision D = classify("std::optional", OnlyOptional);
  EXPECT_EQ(D.Action, AnalyzeAsDecision::Validate);
  EXPECT_EQ(D.Check, "optional-check");

  Enabled OnlyVector{{"vector-check"}};
  D = classify("std::vector", OnlyVector);
  EXPECT_EQ(D.Action, AnalyzeAsDecision::Validate);
  EXPECT_EQ(D.Check, "vector-check");
}

TEST(AnalyzeAsClassify, BothChecksEnabledValidatesBoth) {
  Enabled Both{{"optional-check", "vector-check"}};
  EXPECT_EQ(classify("std::optional", Both).Action,
            AnalyzeAsDecision::Validate);
  EXPECT_EQ(classify("std::vector", Both).Action, AnalyzeAsDecision::Validate);
}

// An unsupported class is reported under whichever analyze_as check is
// enabled, so it can be suppressed like that check's other warnings.
TEST(AnalyzeAsClassify, UnsupportedClassIsReportedUnderAnEnabledCheck) {
  Enabled OnlyOptional{{"optional-check"}};
  AnalyzeAsDecision D = classify("std::vectr", OnlyOptional);
  EXPECT_EQ(D.Action, AnalyzeAsDecision::ReportUnknown);
  EXPECT_EQ(D.Check, "optional-check");

  Enabled OnlyVector{{"vector-check"}};
  D = classify("std::optinal", OnlyVector);
  EXPECT_EQ(D.Action, AnalyzeAsDecision::ReportUnknown);
  EXPECT_EQ(D.Check, "vector-check");
}

TEST(AnalyzeAsClassify, UnsupportedClassIsReportedOnceUnderTheFirstCheck) {
  Enabled Both{{"optional-check", "vector-check"}};
  AnalyzeAsDecision D = classify("std::vectr", Both);
  EXPECT_EQ(D.Action, AnalyzeAsDecision::ReportUnknown);
  EXPECT_EQ(D.Check, "optional-check"); // table order
}

// With no analyze_as check enabled, annotations are irrelevant to the run.
TEST(AnalyzeAsClassify, NothingEnabledIsSilent) {
  Enabled None{{"some-unrelated-check"}};
  EXPECT_EQ(classify("std::optional", None).Action, AnalyzeAsDecision::Silent);
  EXPECT_EQ(classify("std::vector", None).Action, AnalyzeAsDecision::Silent);
  EXPECT_EQ(classify("std::vectr", None).Action, AnalyzeAsDecision::Silent);
}

TEST(AnalyzeAsClassify, NamesMustMatchExactly) {
  Enabled Both{{"optional-check", "vector-check"}};
  EXPECT_EQ(classify("optional", Both).Action,
            AnalyzeAsDecision::ReportUnknown); // missing std::
  EXPECT_EQ(classify("::std::optional", Both).Action,
            AnalyzeAsDecision::ReportUnknown); // leading ::
  EXPECT_EQ(classify("", Both).Action, AnalyzeAsDecision::ReportUnknown);
}

} // namespace
} // namespace clang::tidy
