//===- unittests/Analysis/FlowSensitive/ASTOpsTest.cpp --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/FlowSensitive/ASTOps.h"
#include "TestingSupport.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <memory>

namespace {

using namespace clang;
using namespace dataflow;

using ast_matchers::cxxRecordDecl;
using ast_matchers::hasName;
using ast_matchers::hasType;
using ast_matchers::initListExpr;
using ast_matchers::match;
using ast_matchers::selectFirst;
using test::findValueDecl;
using testing::IsEmpty;
using testing::UnorderedElementsAre;

TEST(ASTOpsTest, RecordInitListHelperOnEmptyUnionInitList) {
  // This is a regression test: The `RecordInitListHelper` used to assert-fail
  // when called for the `InitListExpr` of an empty union.
  std::string Code = R"cc(
    struct S {
      S() : UField{} {};

      union U {} UField;
    };
  )cc";
  std::unique_ptr<ASTUnit> Unit =
      tooling::buildASTFromCodeWithArgs(Code, {"-fsyntax-only", "-std=c++17"});
  auto &ASTCtx = Unit->getASTContext();

  ASSERT_EQ(ASTCtx.getDiagnostics().getClient()->getNumErrors(), 0U);

  auto *InitList = selectFirst<InitListExpr>(
      "init",
      match(initListExpr(hasType(cxxRecordDecl(hasName("U")))).bind("init"),
            ASTCtx));
  ASSERT_NE(InitList, nullptr);

  RecordInitListHelper Helper(InitList);
  EXPECT_THAT(Helper.base_inits(), IsEmpty());
  EXPECT_THAT(Helper.field_inits(), IsEmpty());
}

TEST(ASTOpsTest, ReferencedDeclsOnUnionInitList) {
  // This is a regression test: `getReferencedDecls()` used to return a null
  // `FieldDecl` in this case (in addition to the correct non-null `FieldDecl`)
  // because `getInitializedFieldInUnion()` returns null for the syntactic form
  // of the `InitListExpr`.
  std::string Code = R"cc(
    struct S {
      S() : UField{0} {};

      union U {
        int I;
      } UField;
    };
  )cc";
  std::unique_ptr<ASTUnit> Unit =
      tooling::buildASTFromCodeWithArgs(Code, {"-fsyntax-only", "-std=c++17"});
  auto &ASTCtx = Unit->getASTContext();

  ASSERT_EQ(ASTCtx.getDiagnostics().getClient()->getNumErrors(), 0U);

  auto *InitList = selectFirst<InitListExpr>(
      "init",
      match(initListExpr(hasType(cxxRecordDecl(hasName("U")))).bind("init"),
            ASTCtx));
  ASSERT_NE(InitList, nullptr);
  auto *IDecl = cast<FieldDecl>(findValueDecl(ASTCtx, "I"));

  EXPECT_THAT(getReferencedDecls(*InitList).Fields,
              UnorderedElementsAre(IDecl));
}

TEST(ASTOpsTest, ReferencedDeclsLocalsNotParamsOrStatics) {
  std::string Code = R"cc(
    void func(int Param) {
      static int Static = 0;
      int Local = Param;
      Local = Static;
    }
  )cc";
  std::unique_ptr<ASTUnit> Unit =
      tooling::buildASTFromCodeWithArgs(Code, {"-fsyntax-only", "-std=c++17"});
  auto &ASTCtx = Unit->getASTContext();

  ASSERT_EQ(ASTCtx.getDiagnostics().getClient()->getNumErrors(), 0U);

  auto *Func = cast<FunctionDecl>(findValueDecl(ASTCtx, "func"));
  ASSERT_NE(Func, nullptr);
  auto *LocalDecl = cast<VarDecl>(findValueDecl(ASTCtx, "Local"));

  EXPECT_THAT(getReferencedDecls(*Func).Locals,
              UnorderedElementsAre(LocalDecl));
}

} // namespace
