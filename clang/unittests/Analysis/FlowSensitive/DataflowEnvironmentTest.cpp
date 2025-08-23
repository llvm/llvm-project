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
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Stmt.h"
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
#include <string>

namespace {

using namespace clang;
using namespace dataflow;
using ::clang::dataflow::test::findValueDecl;
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

TEST_F(EnvironmentTest, DifferentReferenceLocInJoin) {
  // This tests the case where the storage location for a reference-type
  // variable is different for two states being joined. We used to believe this
  // could not happen and therefore had an assertion disallowing this; this test
  // exists to demonstrate that we can handle this condition without a failing
  // assertion. See also the discussion here:
  // https://discourse.llvm.org/t/70086/6

  using namespace ast_matchers;

  std::string Code = R"cc(
    void f(int &ref) {}
  )cc";

  auto Unit =
      tooling::buildASTFromCodeWithArgs(Code, {"-fsyntax-only", "-std=c++11"});
  auto &Context = Unit->getASTContext();

  ASSERT_EQ(Context.getDiagnostics().getClient()->getNumErrors(), 0U);

  const ValueDecl *Ref = findValueDecl(Context, "ref");

  Environment Env1(DAContext);
  StorageLocation &Loc1 = Env1.createStorageLocation(Context.IntTy);
  Env1.setStorageLocation(*Ref, Loc1);

  Environment Env2(DAContext);
  StorageLocation &Loc2 = Env2.createStorageLocation(Context.IntTy);
  Env2.setStorageLocation(*Ref, Loc2);

  EXPECT_NE(&Loc1, &Loc2);

  Environment::ValueModel Model;
  Environment EnvJoined =
      Environment::join(Env1, Env2, Model, Environment::DiscardExprState);

  // Joining environments with different storage locations for the same
  // declaration results in the declaration being removed from the joined
  // environment.
  EXPECT_EQ(EnvJoined.getStorageLocation(*Ref), nullptr);
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
  EXPECT_THAT(DAContext.getModeledFields(Context.getCanonicalTagType(Struct)),
              Contains(Member));
}

// This is a repro of a failure case seen in the wild.
TEST_F(EnvironmentTest, CXXDefaultInitExprResultObjIsWrappedExprResultObj) {
  using namespace ast_matchers;

  std::string Code = R"cc(
      struct Inner {};

      struct S {
        S() {}

        Inner i = {};
      };
  )cc";

  auto Unit =
      tooling::buildASTFromCodeWithArgs(Code, {"-fsyntax-only", "-std=c++11"});
  auto &Context = Unit->getASTContext();

  ASSERT_EQ(Context.getDiagnostics().getClient()->getNumErrors(), 0U);

  auto Results =
      match(cxxConstructorDecl(
                hasAnyConstructorInitializer(cxxCtorInitializer(
                    withInitializer(expr().bind("default_init_expr")))))
                .bind("ctor"),
            Context);
  const auto *Constructor = selectFirst<CXXConstructorDecl>("ctor", Results);
  const auto *DefaultInit =
      selectFirst<CXXDefaultInitExpr>("default_init_expr", Results);

  Environment Env(DAContext, *Constructor);
  Env.initialize();
  EXPECT_EQ(&Env.getResultObjectLocation(*DefaultInit),
            &Env.getResultObjectLocation(*DefaultInit->getExpr()));
}

// This test verifies the behavior of `getResultObjectLocation()` in
// scenarios involving inherited constructors.
// Since the specific AST node of interest `CXXConstructorDecl` is implicitly
// generated, we cannot annotate any statements inside of it as we do in tests
// within TransferTest. Thus, the only way to get the right `Environment` is by
// explicitly initializing it as we do in tests within EnvironmentTest.
// This is why this test is not inside TransferTest, where most of the tests for
// `getResultObjectLocation()` are located.
TEST_F(EnvironmentTest, ResultObjectLocationForInheritedCtorInitExpr) {
  using namespace ast_matchers;

  std::string Code = R"(
    struct Base {
      Base(int b) {}
    };
    struct Derived : Base {
      using Base::Base;
    };

    Derived d = Derived(0);
  )";

  auto Unit =
      tooling::buildASTFromCodeWithArgs(Code, {"-fsyntax-only", "-std=c++20"});
  auto &Context = Unit->getASTContext();

  ASSERT_EQ(Context.getDiagnostics().getClient()->getNumErrors(), 0U);

  auto Results =
      match(cxxConstructorDecl(
                hasAnyConstructorInitializer(cxxCtorInitializer(
                    withInitializer(expr().bind("inherited_ctor_init_expr")))))
                .bind("ctor"),
            Context);
  const auto *Constructor = selectFirst<CXXConstructorDecl>("ctor", Results);
  const auto *InheritedCtorInit = selectFirst<CXXInheritedCtorInitExpr>(
      "inherited_ctor_init_expr", Results);

  EXPECT_EQ(InheritedCtorInit->child_begin(), InheritedCtorInit->child_end());

  Environment Env(DAContext, *Constructor);
  Env.initialize();

  RecordStorageLocation &Loc = Env.getResultObjectLocation(*InheritedCtorInit);
  EXPECT_NE(&Loc, nullptr);

  EXPECT_EQ(&Loc, Env.getThisPointeeStorageLocation());
}

TEST_F(EnvironmentTest, Stmt) {
  using namespace ast_matchers;

  std::string Code = R"cc(
      struct S { int i; };
      void foo() {
        S AnS = S{1};
      }
    )cc";
  auto Unit =
      tooling::buildASTFromCodeWithArgs(Code, {"-fsyntax-only", "-std=c++11"});
  auto &Context = Unit->getASTContext();

  ASSERT_EQ(Context.getDiagnostics().getClient()->getNumErrors(), 0U);

  auto *DeclStatement = const_cast<DeclStmt *>(selectFirst<DeclStmt>(
      "d", match(declStmt(hasSingleDecl(varDecl(hasName("AnS")))).bind("d"),
                 Context)));
  ASSERT_THAT(DeclStatement, NotNull());
  auto *Init = (cast<VarDecl>(*DeclStatement->decl_begin()))->getInit();
  ASSERT_THAT(Init, NotNull());

  // Verify that we can retrieve the result object location for the initializer
  // expression when we analyze the DeclStmt for `AnS`.
  Environment Env(DAContext, *DeclStatement);
  // Don't crash when initializing.
  Env.initialize();
  // And don't crash when retrieving the result object location.
  Env.getResultObjectLocation(*Init);
}

// This is a crash repro.
TEST_F(EnvironmentTest, LambdaCapturingThisInFieldInitializer) {
  using namespace ast_matchers;
  std::string Code = R"cc(
      struct S {
        int f{[this]() { return 1; }()};
      };
    )cc";

  auto Unit =
      tooling::buildASTFromCodeWithArgs(Code, {"-fsyntax-only", "-std=c++11"});
  auto &Context = Unit->getASTContext();

  ASSERT_EQ(Context.getDiagnostics().getClient()->getNumErrors(), 0U);

  auto *LambdaCallOperator = selectFirst<CXXMethodDecl>(
      "method", match(cxxMethodDecl(hasName("operator()"),
                                    ofClass(cxxRecordDecl(isLambda())))
                          .bind("method"),
                      Context));

  Environment Env(DAContext, *LambdaCallOperator);
  // Don't crash when initializing.
  Env.initialize();
  // And initialize the captured `this` pointee.
  ASSERT_NE(nullptr, Env.getThisPointeeStorageLocation());
}

} // namespace
