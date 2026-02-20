//===- unittests/Analysis/Scalable/SummaryNameTest.cpp --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Scalable/Model/SummaryName.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace ssaf;

namespace {

TEST(SummaryNameTest, Equality) {
  const auto TestAnalysis1 = SummaryName("TestAnalysis1");
  const auto Alternative1 = SummaryName("TestAnalysis1");
  const auto TestAnalysis2 = SummaryName("TestAnalysis2");

  EXPECT_EQ(TestAnalysis1, Alternative1); // Idempotency.
  EXPECT_NE(Alternative1, TestAnalysis2); // Inequality.
}

TEST(SummaryNameTest, LessThan) {
  const auto TestAnalysis1 = SummaryName("TestAnalysis1");
  const auto Alternative1 = SummaryName("TestAnalysis1");

  const auto TestAnalysis2 = SummaryName("TestAnalysis2");
  const auto TestAnalysis3 = SummaryName("TestAnalysis3");

  // Equivalency.
  EXPECT_FALSE(TestAnalysis1 < Alternative1);
  EXPECT_FALSE(Alternative1 < TestAnalysis1);

  // Transitivity.
  EXPECT_LT(TestAnalysis1, TestAnalysis2);
  EXPECT_LT(TestAnalysis2, TestAnalysis3);
  EXPECT_LT(TestAnalysis1, TestAnalysis3);
}

TEST(SummaryNameTest, Str) {
  const auto Handle1 = SummaryName("TestAnalysis1");
  const auto Handle2 = SummaryName("TestAnalysis1");
  const auto Handle3 = SummaryName("TestAnalysis2");

  EXPECT_EQ(Handle1.str(), "TestAnalysis1");
  EXPECT_EQ(Handle2.str(), "TestAnalysis1");
  EXPECT_EQ(Handle3.str(), "TestAnalysis2");
}

TEST(SummaryNameTest, StreamOutput) {
  std::string S;
  llvm::raw_string_ostream(S) << SummaryName("MyAnalysis");
  EXPECT_EQ(S, "SummaryName(MyAnalysis)");
}

} // namespace
