//===- unittests/Analysis/FlowSensitive/DataflowAnalysisContextTest.cpp ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/FlowSensitive/DataflowAnalysisContext.h"
#include "clang/Analysis/FlowSensitive/WatchedLiteralsSolver.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <memory>

namespace {

using namespace clang;
using namespace dataflow;

class DataflowAnalysisContextTest : public ::testing::Test {
protected:
  DataflowAnalysisContextTest()
      : Context(std::make_unique<WatchedLiteralsSolver>()) {}

  DataflowAnalysisContext Context;
};

TEST_F(DataflowAnalysisContextTest,
       CreateAtomicBoolValueReturnsDistinctValues) {
  auto &X = Context.createAtomicBoolValue();
  auto &Y = Context.createAtomicBoolValue();
  EXPECT_NE(&X, &Y);
}

TEST_F(DataflowAnalysisContextTest,
       GetOrCreateConjunctionValueReturnsSameExprGivenSameArgs) {
  auto &X = Context.createAtomicBoolValue();
  auto &XAndX = Context.getOrCreateConjunctionValue(X, X);
  EXPECT_EQ(&XAndX, &X);
}

TEST_F(DataflowAnalysisContextTest,
       GetOrCreateConjunctionValueReturnsSameExprOnSubsequentCalls) {
  auto &X = Context.createAtomicBoolValue();
  auto &Y = Context.createAtomicBoolValue();
  auto &XAndY1 = Context.getOrCreateConjunctionValue(X, Y);
  auto &XAndY2 = Context.getOrCreateConjunctionValue(X, Y);
  EXPECT_EQ(&XAndY1, &XAndY2);

  auto &YAndX = Context.getOrCreateConjunctionValue(Y, X);
  EXPECT_EQ(&XAndY1, &YAndX);

  auto &Z = Context.createAtomicBoolValue();
  auto &XAndZ = Context.getOrCreateConjunctionValue(X, Z);
  EXPECT_NE(&XAndY1, &XAndZ);
}

TEST_F(DataflowAnalysisContextTest,
       GetOrCreateDisjunctionValueReturnsSameExprGivenSameArgs) {
  auto &X = Context.createAtomicBoolValue();
  auto &XOrX = Context.getOrCreateDisjunctionValue(X, X);
  EXPECT_EQ(&XOrX, &X);
}

TEST_F(DataflowAnalysisContextTest,
       GetOrCreateDisjunctionValueReturnsSameExprOnSubsequentCalls) {
  auto &X = Context.createAtomicBoolValue();
  auto &Y = Context.createAtomicBoolValue();
  auto &XOrY1 = Context.getOrCreateDisjunctionValue(X, Y);
  auto &XOrY2 = Context.getOrCreateDisjunctionValue(X, Y);
  EXPECT_EQ(&XOrY1, &XOrY2);

  auto &YOrX = Context.getOrCreateDisjunctionValue(Y, X);
  EXPECT_EQ(&XOrY1, &YOrX);

  auto &Z = Context.createAtomicBoolValue();
  auto &XOrZ = Context.getOrCreateDisjunctionValue(X, Z);
  EXPECT_NE(&XOrY1, &XOrZ);
}

TEST_F(DataflowAnalysisContextTest,
       GetOrCreateNegationValueReturnsSameExprOnSubsequentCalls) {
  auto &X = Context.createAtomicBoolValue();
  auto &NotX1 = Context.getOrCreateNegationValue(X);
  auto &NotX2 = Context.getOrCreateNegationValue(X);
  EXPECT_EQ(&NotX1, &NotX2);

  auto &Y = Context.createAtomicBoolValue();
  auto &NotY = Context.getOrCreateNegationValue(Y);
  EXPECT_NE(&NotX1, &NotY);
}

} // namespace
