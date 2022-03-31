//===- unittests/Analysis/FlowSensitive/DataflowEnvironmentTest.cpp -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "NoopAnalysis.h"
#include "TestingSupport.h"
#include "clang/Analysis/FlowSensitive/DataflowAnalysisContext.h"
#include "clang/Analysis/FlowSensitive/Value.h"
#include "clang/Analysis/FlowSensitive/WatchedLiteralsSolver.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <memory>

namespace {

using namespace clang;
using namespace dataflow;
using ::testing::ElementsAre;
using ::testing::Pair;

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

TEST_F(EnvironmentTest, CreateValueRecursiveType) {
  using namespace ast_matchers;

  std::string Code = R"cc(
    struct Recursive {
      bool X;
      Recursive *R;
    };
  )cc";

  auto Unit =
      tooling::buildASTFromCodeWithArgs(Code, {"-fsyntax-only", "-std=c++11"});
  auto &Context = Unit->getASTContext();

  ASSERT_EQ(Context.getDiagnostics().getClient()->getNumErrors(), 0U);

  auto Results =
      match(qualType(hasDeclaration(recordDecl(
                         hasName("Recursive"),
                         has(fieldDecl(hasName("R")).bind("field-r")))))
                .bind("target"),
            Context);
  const QualType *Ty = selectFirst<QualType>("target", Results);
  const FieldDecl *R = selectFirst<FieldDecl>("field-r", Results);
  ASSERT_TRUE(Ty != nullptr && !Ty->isNull());
  ASSERT_TRUE(R != nullptr);

  // Verify that the struct and the field (`R`) with first appearance of the
  // type is created successfully.
  Value *Val = Env.createValue(*Ty);
  ASSERT_NE(Val, nullptr);
  StructValue *SVal = clang::dyn_cast<StructValue>(Val);
  ASSERT_NE(SVal, nullptr);
  Val = SVal->getChild(*R);
  ASSERT_NE(Val, nullptr);
  PointerValue *PV = clang::dyn_cast<PointerValue>(Val);
  EXPECT_NE(PV, nullptr);
}

} // namespace
