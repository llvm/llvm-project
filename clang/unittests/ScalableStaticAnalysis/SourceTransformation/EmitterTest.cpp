//===- EmitterTest.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/Sarif.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/ScalableStaticAnalysis/SourceTransformation/SourceEditEmitter.h"
#include "clang/ScalableStaticAnalysis/SourceTransformation/TransformationReportEmitter.h"
#include "gtest/gtest.h"
#include <vector>

using namespace llvm;
using namespace clang;
using namespace ssaf;

namespace {

class RecordingEditEmitter : public SourceEditEmitter {
public:
  std::vector<clang::tooling::Replacement> Replacements;

  void addReplacement(clang::tooling::Replacement R) override {
    Replacements.push_back(std::move(R));
  }
};

class RecordingReportEmitter : public TransformationReportEmitter {
public:
  struct Entry {
    std::string RuleId;
    clang::SarifResultLevel Level;
    clang::CharSourceRange Range;
    std::string Message;
  };
  std::vector<Entry> Results;

  void addResult(StringRef RuleId, clang::SarifResultLevel Level,
                 clang::CharSourceRange Range, StringRef Message) override {
    Results.push_back({RuleId.str(), Level, Range, Message.str()});
  }
};

TEST(SourceEditEmitterTest, AccumulatesInOrder) {
  RecordingEditEmitter E;
  E.addReplacement(clang::tooling::Replacement("a.cpp", 0, 0, "// 1"));
  E.addReplacement(clang::tooling::Replacement("a.cpp", 10, 0, "// 2"));
  ASSERT_EQ(E.Replacements.size(), 2u);
  EXPECT_EQ(E.Replacements[0].getReplacementText(), "// 1");
  EXPECT_EQ(E.Replacements[1].getReplacementText(), "// 2");
  EXPECT_EQ(E.Replacements[0].getOffset(), 0u);
  EXPECT_EQ(E.Replacements[1].getOffset(), 10u);
}

TEST(TransformationReportEmitterTest, AccumulatesInOrder) {
  RecordingReportEmitter R;
  R.addResult("rule-a", clang::SarifResultLevel::Note, clang::CharSourceRange{},
              "first");
  R.addResult("rule-b", clang::SarifResultLevel::Warning,
              clang::CharSourceRange{}, "second");
  ASSERT_EQ(R.Results.size(), 2u);
  EXPECT_EQ(R.Results[0].RuleId, "rule-a");
  EXPECT_EQ(R.Results[0].Level, clang::SarifResultLevel::Note);
  EXPECT_EQ(R.Results[0].Message, "first");
  EXPECT_EQ(R.Results[1].RuleId, "rule-b");
  EXPECT_EQ(R.Results[1].Level, clang::SarifResultLevel::Warning);
  EXPECT_EQ(R.Results[1].Message, "second");
}

TEST(TransformationReportEmitterTest, AcceptsInvalidRange) {
  RecordingReportEmitter R;
  R.addResult("rule", clang::SarifResultLevel::Note, clang::CharSourceRange{},
              "no-location");
  ASSERT_EQ(R.Results.size(), 1u);
  EXPECT_FALSE(R.Results[0].Range.isValid());
}

} // namespace
