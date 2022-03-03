//===- unittests/Analysis/FlowSensitive/DataflowEnvironmentTest.cpp -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/DataflowAnalysisContext.h"
#include "clang/Analysis/FlowSensitive/WatchedLiteralsSolver.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <memory>

namespace {

using namespace clang;
using namespace dataflow;

class EnvironmentTest : public ::testing::Test {
  DataflowAnalysisContext Context;

protected:
  EnvironmentTest()
      : Context(std::make_unique<WatchedLiteralsSolver>()), Env(Context) {}

  Environment Env;
};

TEST_F(EnvironmentTest, MakeImplicationReturnsTrueGivenSameArgs) {
  auto &X = Env.makeAtomicBoolValue();
  auto &XEqX = Env.makeImplication(X, X);
  EXPECT_EQ(&XEqX, &Env.getBoolLiteralValue(true));
}

TEST_F(EnvironmentTest, MakeIffReturnsTrueGivenSameArgs) {
  auto &X = Env.makeAtomicBoolValue();
  auto &XEqX = Env.makeIff(X, X);
  EXPECT_EQ(&XEqX, &Env.getBoolLiteralValue(true));
}

TEST_F(EnvironmentTest, FlowCondition) {
  EXPECT_TRUE(Env.flowConditionImplies(Env.getBoolLiteralValue(true)));
  EXPECT_FALSE(Env.flowConditionImplies(Env.getBoolLiteralValue(false)));

  auto &X = Env.makeAtomicBoolValue();
  EXPECT_FALSE(Env.flowConditionImplies(X));

  Env.addToFlowCondition(X);
  EXPECT_TRUE(Env.flowConditionImplies(X));

  auto &NotX = Env.makeNot(X);
  EXPECT_FALSE(Env.flowConditionImplies(NotX));
}

} // namespace
