//===- unittests/Analysis/FlowSensitive/DataflowEnvironmentTest.cpp -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "TestingSupport.h"
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
using ::clang::dataflow::test::getFieldValue;
using ::testing::Contains;
using ::testing::IsNull;
using ::testing::NotNull;

class EnvironmentTest : public ::testing::Test {
protected:
  EnvironmentTest() : DAContext(std::make_unique<WatchedLiteralsSolver>()) {}

  DataflowAnalysisContext DAContext;
};

TEST_F(EnvironmentTest, FlowCondition) {
  Environment Env(DAContext);
  auto &A = Env.arena();

  EXPECT_TRUE(Env.proves(A.makeLiteral(true)));
  EXPECT_TRUE(Env.allows(A.makeLiteral(true)));
  EXPECT_FALSE(Env.proves(A.makeLiteral(false)));
  EXPECT_FALSE(Env.allows(A.makeLiteral(false)));

  auto &X = A.makeAtomRef(A.makeAtom());
  EXPECT_FALSE(Env.proves(X));
  EXPECT_TRUE(Env.allows(X));

  Env.assume(X);
  EXPECT_TRUE(Env.proves(X));
  EXPECT_TRUE(Env.allows(X));

  auto &NotX = A.makeNot(X);
  EXPECT_FALSE(Env.proves(NotX));
  EXPECT_FALSE(Env.allows(NotX));
}

TEST_F(EnvironmentTest, SetAndGetValueOnCfgOmittedNodes) {
  // Check that we can set a value on an expression that is omitted from the CFG
  // (see `ignoreCFGOmittedNodes()`), then retrieve that same value from the
  // expression. This is a regression test; `setValue()` and `getValue()`
  // previously did not use `ignoreCFGOmittedNodes()` consistently.

  using namespace ast_matchers;

  std::string Code = R"cc(
    struct S {
      int f();
    };
    void target() {
      // Method call on a temporary produces an `ExprWithCleanups`.
      S().f();
      (1);
    }
  )cc";

  auto Unit =
      tooling::buildASTFromCodeWithArgs(Code, {"-fsyntax-only", "-std=c++17"});
  auto &Context = Unit->getASTContext();

  ASSERT_EQ(Context.getDiagnostics().getClient()->getNumErrors(), 0U);

  const ExprWithCleanups *WithCleanups = selectFirst<ExprWithCleanups>(
      "cleanups",
      match(exprWithCleanups(hasType(isInteger())).bind("cleanups"), Context));
  ASSERT_NE(WithCleanups, nullptr);

  const ParenExpr *Paren = selectFirst<ParenExpr>(
      "paren", match(parenExpr(hasType(isInteger())).bind("paren"), Context));
  ASSERT_NE(Paren, nullptr);

  Environment Env(DAContext);
  IntegerValue *Val1 =
      cast<IntegerValue>(Env.createValue(Unit->getASTContext().IntTy));
  Env.setValue(*WithCleanups, *Val1);
  EXPECT_EQ(Env.getValue(*WithCleanups), Val1);

  IntegerValue *Val2 =
      cast<IntegerValue>(Env.createValue(Unit->getASTContext().IntTy));
  Env.setValue(*Paren, *Val2);
  EXPECT_EQ(Env.getValue(*Paren), Val2);
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
  Env.initialize();
  auto &SLoc = cast<RecordStorageLocation>(Env.createObject(Ty));
  PointerValue *PV = cast_or_null<PointerValue>(getFieldValue(&SLoc, *R, Env));
  EXPECT_THAT(PV, NotNull());
}

TEST_F(EnvironmentTest, JoinRecords) {
  using namespace ast_matchers;

  std::string Code = R"cc(
    struct S {};
    // Need to use the type somewhere so that the `QualType` gets created;
    S s;
  )cc";

  auto Unit =
      tooling::buildASTFromCodeWithArgs(Code, {"-fsyntax-only", "-std=c++11"});
  auto &Context = Unit->getASTContext();

  ASSERT_EQ(Context.getDiagnostics().getClient()->getNumErrors(), 0U);

  auto Results =
      match(qualType(hasDeclaration(recordDecl(hasName("S")))).bind("SType"),
            Context);
  const QualType *TyPtr = selectFirst<QualType>("SType", Results);
  ASSERT_THAT(TyPtr, NotNull());
  QualType Ty = *TyPtr;
  ASSERT_FALSE(Ty.isNull());

  auto *ConstructExpr = CXXConstructExpr::CreateEmpty(Context, 0);
  ConstructExpr->setType(Ty);
  ConstructExpr->setValueKind(VK_PRValue);

  // Two different `RecordValue`s with the same location are joined into a
  // third `RecordValue` with that same location.
  {
    Environment Env1(DAContext);
    auto &Val1 = *cast<RecordValue>(Env1.createValue(Ty));
    RecordStorageLocation &Loc = Val1.getLoc();
    Env1.setValue(Loc, Val1);

    Environment Env2(DAContext);
    auto &Val2 = Env2.create<RecordValue>(Loc);
    Env2.setValue(Loc, Val2);
    Env2.setValue(Loc, Val2);

    Environment::ValueModel Model;
    Environment EnvJoined =
        Environment::join(Env1, Env2, Model, Environment::DiscardExprState);
    auto *JoinedVal = cast<RecordValue>(EnvJoined.getValue(Loc));
    EXPECT_NE(JoinedVal, &Val1);
    EXPECT_NE(JoinedVal, &Val2);
    EXPECT_EQ(&JoinedVal->getLoc(), &Loc);
  }
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
  Env.initialize();
  EXPECT_THAT(Env.getValue(*Var), NotNull());
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
  Env.initialize();
  auto &Loc = cast<RecordStorageLocation>(Env.createObject(QTy));
  EXPECT_THAT(getFieldValue(&Loc, *XDecl, Env), NotNull());
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
  Env.initialize();
  const auto *GlobalLoc =
      cast<RecordStorageLocation>(Env.getStorageLocation(*GlobalDecl));
  auto *BarVal = getFieldValue(GlobalLoc, *BarDecl, Env);
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
  Env.initialize();
  EXPECT_THAT(Env.getValue(*Var), NotNull());
}

// Pointers to Members are a tricky case of accessor calls, complicated further
// when using templates where the pointer to the member is a template argument.
// This is a repro of a failure case seen in the wild.
TEST_F(EnvironmentTest,
       ModelMemberForAccessorUsingMethodPointerThroughTemplate) {
  using namespace ast_matchers;

  std::string Code = R"cc(
      struct S {
        int accessor() {return member;}

        int member = 0;
      };

      template <auto method>
      int Target(S* S) {
        return (S->*method)();
      }

     // We want to analyze the instantiation of Target for the accessor.
     int Instantiator () {S S; return Target<&S::accessor>(&S); }
  )cc";

  auto Unit =
      // C++17 for the simplifying use of auto in the template declaration.
      tooling::buildASTFromCodeWithArgs(Code, {"-fsyntax-only", "-std=c++17"});
  auto &Context = Unit->getASTContext();

  ASSERT_EQ(Context.getDiagnostics().getClient()->getNumErrors(), 0U);

  auto Results = match(
      decl(anyOf(functionDecl(hasName("Target"), isTemplateInstantiation())
                     .bind("target"),
                 fieldDecl(hasName("member")).bind("member"),
                 recordDecl(hasName("S")).bind("struct"))),
      Context);
  const auto *Fun = selectFirst<FunctionDecl>("target", Results);
  const auto *Struct = selectFirst<RecordDecl>("struct", Results);
  const auto *Member = selectFirst<FieldDecl>("member", Results);
  ASSERT_THAT(Fun, NotNull());
  ASSERT_THAT(Struct, NotNull());
  ASSERT_THAT(Member, NotNull());

  // Verify that `member` is modeled for `S` when we analyze
  // `Target<&S::accessor>`.
  Environment Env(DAContext, *Fun);
  Env.initialize();
  EXPECT_THAT(DAContext.getModeledFields(QualType(Struct->getTypeForDecl(), 0)),
              Contains(Member));
}

TEST_F(EnvironmentTest, RefreshRecordValue) {
  using namespace ast_matchers;

  std::string Code = R"cc(
     struct S {};
     void target () {
       S s;
       s;
     }
  )cc";

  auto Unit =
      tooling::buildASTFromCodeWithArgs(Code, {"-fsyntax-only", "-std=c++11"});
  auto &Context = Unit->getASTContext();

  ASSERT_EQ(Context.getDiagnostics().getClient()->getNumErrors(), 0U);

  auto Results = match(functionDecl(hasName("target")).bind("target"), Context);
  const auto *Target = selectFirst<FunctionDecl>("target", Results);
  ASSERT_THAT(Target, NotNull());

  Results = match(declRefExpr(to(varDecl(hasName("s")))).bind("s"), Context);
  const auto *DRE = selectFirst<DeclRefExpr>("s", Results);
  ASSERT_THAT(DRE, NotNull());

  Environment Env(DAContext, *Target);
  EXPECT_THAT(Env.getStorageLocation(*DRE), IsNull());
  refreshRecordValue(*DRE, Env);
  EXPECT_THAT(Env.getStorageLocation(*DRE), NotNull());
}

} // namespace
