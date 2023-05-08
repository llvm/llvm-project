//===- unittests/Analysis/FlowSensitive/DataflowEnvironmentTest.cpp -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/AST/DeclCXX.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/FlowSensitive/DataflowAnalysisContext.h"
#include "clang/Analysis/FlowSensitive/StorageLocation.h"
#include "clang/Analysis/FlowSensitive/Value.h"
#include "clang/Analysis/FlowSensitive/WatchedLiteralsSolver.h"
#include "clang/Tooling/Tooling.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <memory>

namespace {

using namespace clang;
using namespace dataflow;
using ::testing::ElementsAre;
using ::testing::NotNull;
using ::testing::Pair;

class EnvironmentTest : public ::testing::Test {
protected:
  EnvironmentTest() : DAContext(std::make_unique<WatchedLiteralsSolver>()) {}

  DataflowAnalysisContext DAContext;
};

TEST_F(EnvironmentTest, FlowCondition) {
  Environment Env(DAContext);

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
    // Use both fields to force them to be created with `createValue`.
    void Usage(Recursive R) { (void)R.X; (void)R.R; }
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
  const QualType *TyPtr = selectFirst<QualType>("target", Results);
  ASSERT_THAT(TyPtr, NotNull());
  QualType Ty = *TyPtr;
  ASSERT_FALSE(Ty.isNull());

  const FieldDecl *R = selectFirst<FieldDecl>("field-r", Results);
  ASSERT_THAT(R, NotNull());

  Results = match(functionDecl(hasName("Usage")).bind("fun"), Context);
  const auto *Fun = selectFirst<FunctionDecl>("fun", Results);
  ASSERT_THAT(Fun, NotNull());

  // Verify that the struct and the field (`R`) with first appearance of the
  // type is created successfully.
  Environment Env(DAContext, *Fun);
  Value *Val = Env.createValue(Ty);
  ASSERT_NE(Val, nullptr);
  StructValue *SVal = clang::dyn_cast<StructValue>(Val);
  ASSERT_NE(SVal, nullptr);
  Val = SVal->getChild(*R);
  ASSERT_NE(Val, nullptr);
  PointerValue *PV = clang::dyn_cast<PointerValue>(Val);
  EXPECT_NE(PV, nullptr);
}

TEST_F(EnvironmentTest, InitGlobalVarsFun) {
  using namespace ast_matchers;

  std::string Code = R"cc(
     int Global = 0;
     int Target () { return Global; }
  )cc";

  auto Unit =
      tooling::buildASTFromCodeWithArgs(Code, {"-fsyntax-only", "-std=c++11"});
  auto &Context = Unit->getASTContext();

  ASSERT_EQ(Context.getDiagnostics().getClient()->getNumErrors(), 0U);

  auto Results =
      match(decl(anyOf(varDecl(hasName("Global")).bind("global"),
                       functionDecl(hasName("Target")).bind("target"))),
            Context);
  const auto *Fun = selectFirst<FunctionDecl>("target", Results);
  const auto *Var = selectFirst<VarDecl>("global", Results);
  ASSERT_THAT(Fun, NotNull());
  ASSERT_THAT(Var, NotNull());

  // Verify the global variable is populated when we analyze `Target`.
  Environment Env(DAContext, *Fun);
  EXPECT_THAT(Env.getValue(*Var, SkipPast::None), NotNull());
}

// Tests that fields mentioned only in default member initializers are included
// in the set of tracked fields.
TEST_F(EnvironmentTest, IncludeFieldsFromDefaultInitializers) {
  using namespace ast_matchers;

  std::string Code = R"cc(
     struct S {
       S() {}
       int X = 3;
       int Y = X;
     };
     S foo();
  )cc";

  auto Unit =
      tooling::buildASTFromCodeWithArgs(Code, {"-fsyntax-only", "-std=c++11"});
  auto &Context = Unit->getASTContext();

  ASSERT_EQ(Context.getDiagnostics().getClient()->getNumErrors(), 0U);

  auto Results = match(
      qualType(hasDeclaration(
                   cxxRecordDecl(hasName("S"),
                                 hasMethod(cxxConstructorDecl().bind("target")))
                       .bind("struct")))
          .bind("ty"),
      Context);
  const auto *Constructor = selectFirst<FunctionDecl>("target", Results);
  const auto *Rec = selectFirst<RecordDecl>("struct", Results);
  const auto QTy = *selectFirst<QualType>("ty", Results);
  ASSERT_THAT(Constructor, NotNull());
  ASSERT_THAT(Rec, NotNull());
  ASSERT_FALSE(QTy.isNull());

  auto Fields = Rec->fields();
  FieldDecl *XDecl = nullptr;
  for (FieldDecl *Field : Fields) {
    if (Field->getNameAsString() == "X") {
      XDecl = Field;
      break;
    }
  }
  ASSERT_THAT(XDecl, NotNull());

  // Verify that the `X` field of `S` is populated when analyzing the
  // constructor, even though it is not referenced directly in the constructor.
  Environment Env(DAContext, *Constructor);
  auto *Val = cast<StructValue>(Env.createValue(QTy));
  ASSERT_THAT(Val, NotNull());
  EXPECT_THAT(Val->getChild(*XDecl), NotNull());
}

TEST_F(EnvironmentTest, InitGlobalVarsFieldFun) {
  using namespace ast_matchers;

  std::string Code = R"cc(
     struct S { int Bar; };
     S Global = {0};
     int Target () { return Global.Bar; }
  )cc";

  auto Unit =
      tooling::buildASTFromCodeWithArgs(Code, {"-fsyntax-only", "-std=c++11"});
  auto &Context = Unit->getASTContext();

  ASSERT_EQ(Context.getDiagnostics().getClient()->getNumErrors(), 0U);

  auto Results =
      match(decl(anyOf(varDecl(hasName("Global")).bind("global"),
                       functionDecl(hasName("Target")).bind("target"))),
            Context);
  const auto *Fun = selectFirst<FunctionDecl>("target", Results);
  const auto *GlobalDecl = selectFirst<VarDecl>("global", Results);
  ASSERT_THAT(Fun, NotNull());
  ASSERT_THAT(GlobalDecl, NotNull());

  ASSERT_TRUE(GlobalDecl->getType()->isStructureType());
  auto GlobalFields = GlobalDecl->getType()->getAsRecordDecl()->fields();

  FieldDecl *BarDecl = nullptr;
  for (FieldDecl *Field : GlobalFields) {
    if (Field->getNameAsString() == "Bar") {
      BarDecl = Field;
      break;
    }
    FAIL() << "Unexpected field: " << Field->getNameAsString();
  }
  ASSERT_THAT(BarDecl, NotNull());

  // Verify the global variable is populated when we analyze `Target`.
  Environment Env(DAContext, *Fun);
  const auto *GlobalLoc =
      cast<AggregateStorageLocation>(Env.getStorageLocation(*GlobalDecl));
  const auto *GlobalVal = cast<StructValue>(Env.getValue(*GlobalLoc));
  const auto *BarVal = GlobalVal->getChild(*BarDecl);
  ASSERT_THAT(BarVal, NotNull());
  EXPECT_TRUE(isa<IntegerValue>(BarVal));
}

TEST_F(EnvironmentTest, InitGlobalVarsConstructor) {
  using namespace ast_matchers;

  std::string Code = R"cc(
     int Global = 0;
     struct Target {
       Target() : Field(Global) {}
       int Field;
     };
  )cc";

  auto Unit =
      tooling::buildASTFromCodeWithArgs(Code, {"-fsyntax-only", "-std=c++11"});
  auto &Context = Unit->getASTContext();

  ASSERT_EQ(Context.getDiagnostics().getClient()->getNumErrors(), 0U);

  auto Results =
      match(decl(anyOf(
                varDecl(hasName("Global")).bind("global"),
                cxxConstructorDecl(ofClass(hasName("Target"))).bind("target"))),
            Context);
  const auto *Ctor = selectFirst<CXXConstructorDecl>("target", Results);
  const auto *Var = selectFirst<VarDecl>("global", Results);
  ASSERT_TRUE(Ctor != nullptr);
  ASSERT_THAT(Var, NotNull());

  // Verify the global variable is populated when we analyze `Target`.
  Environment Env(DAContext, *Ctor);
  EXPECT_THAT(Env.getValue(*Var, SkipPast::None), NotNull());
}

} // namespace
