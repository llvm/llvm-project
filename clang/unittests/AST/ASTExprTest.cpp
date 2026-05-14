//===- unittests/AST/ASTExprTest.cpp --- AST Expr tests -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains tests for AST Expr related methods.
//
//===----------------------------------------------------------------------===//

#include "ASTPrint.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/AST/IgnoreExpr.h"
#include "clang/AST/OpenACCClause.h"
#include "clang/AST/StmtOpenACC.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"

using namespace clang;

using clang::ast_matchers::cxxRecordDecl;
using clang::ast_matchers::forStmt;
using clang::ast_matchers::has;
using clang::ast_matchers::hasName;
using clang::ast_matchers::initListExpr;
using clang::ast_matchers::match;
using clang::ast_matchers::selectFirst;
using clang::ast_matchers::stmt;
using clang::ast_matchers::substNonTypeTemplateParmExpr;
using clang::ast_matchers::varDecl;
using clang::tooling::buildASTFromCode;
using clang::tooling::buildASTFromCodeWithArgs;

static IntegerLiteral *createIntLiteral(ASTContext &Ctx, uint32_t Value) {
  const int numBits = 32;
  return IntegerLiteral::Create(Ctx, llvm::APInt(numBits, Value), Ctx.IntTy,
                                {});
}

const CXXRecordDecl *getCXXRecordDeclNode(ASTUnit *AST,
                                          const std::string &Name) {
  auto Result =
      match(cxxRecordDecl(hasName(Name)).bind("record"), AST->getASTContext());
  EXPECT_FALSE(Result.empty());
  return Result[0].getNodeAs<CXXRecordDecl>("record");
}

const VarDecl *getVariableNode(ASTUnit *AST, const std::string &Name) {
  auto Result = match(varDecl(hasName(Name)).bind("var"), AST->getASTContext());
  EXPECT_EQ(Result.size(), 1u);
  return Result[0].getNodeAs<VarDecl>("var");
}

TEST(ASTExpr, IgnoreExprCallbackForwarded) {
  constexpr char Code[] = "";
  auto AST = tooling::buildASTFromCodeWithArgs(Code, /*Args=*/{"-std=c++20"});
  ASTContext &Ctx = AST->getASTContext();

  struct IgnoreParens {
    Expr *operator()(Expr *E) & { return nullptr; }
    Expr *operator()(Expr *E) && {
      if (auto *PE = dyn_cast<ParenExpr>(E)) {
        return PE->getSubExpr();
      }
      return E;
    }
  };

  {
    auto *IntExpr = createIntLiteral(Ctx, 10);
    ParenExpr *PE =
        new (Ctx) ParenExpr(SourceLocation{}, SourceLocation{}, IntExpr);
    EXPECT_EQ(IntExpr, IgnoreExprNodes(PE, IgnoreParens{}));
  }

  {
    IgnoreParens CB{};
    auto *IntExpr = createIntLiteral(Ctx, 10);
    ParenExpr *PE =
        new (Ctx) ParenExpr(SourceLocation{}, SourceLocation{}, IntExpr);
    EXPECT_EQ(nullptr, IgnoreExprNodes(PE, CB));
  }
}

TEST(ASTExpr, InitListIsConstantInitialized) {
  auto AST = buildASTFromCode(R"cpp(
    struct Empty {};
    struct Foo : Empty { int x, y; };
    int gv;
  )cpp");
  ASTContext &Ctx = AST->getASTContext();
  const CXXRecordDecl *Empty = getCXXRecordDeclNode(AST.get(), "Empty");
  const CXXRecordDecl *Foo = getCXXRecordDeclNode(AST.get(), "Foo");

  SourceLocation Loc{};
  InitListExpr *BaseInit =
      new (Ctx) InitListExpr(Ctx, Loc, {}, Loc, /*isExplicit=*/true);
  BaseInit->setType(Ctx.getCanonicalTagType(Empty));
  Expr *Exprs[3] = {
      BaseInit,
      createIntLiteral(Ctx, 13),
      createIntLiteral(Ctx, 42),
  };
  InitListExpr *FooInit =
      new (Ctx) InitListExpr(Ctx, Loc, Exprs, Loc, /*isExplicit=*/true);
  FooInit->setType(Ctx.getCanonicalTagType(Foo));
  EXPECT_TRUE(FooInit->isConstantInitializer(Ctx, false));

  // Replace the last initializer with something non-constant and make sure
  // this returns false. Previously we had a bug where we didn't count base
  // initializers, and only iterated over fields.
  const VarDecl *GV = getVariableNode(AST.get(), "gv");
  auto *Ref = new (Ctx) DeclRefExpr(Ctx, const_cast<VarDecl *>(GV), false,
                                    Ctx.IntTy, VK_LValue, Loc);
  (void)FooInit->updateInit(Ctx, 2, Ref);
  EXPECT_FALSE(FooInit->isConstantInitializer(Ctx, false));
}

struct NestedInitListParams {
  const char *Code;
  bool InnerIsExplicit;
} NestedInitListParamArray[] = {
    {R"cpp(
        struct Outer {
          struct { int i; } inner;
        } o = {};
      )cpp",
     false},
    {R"cpp(
        struct Outer {
          struct { int i; } inner;
        } o = {1};
      )cpp",
     false},
    {R"cpp(
        struct Outer {
          struct { int i; } inner;
        } o = {{1}};
      )cpp",
     true},
    {R"cpp(
        struct Outer {
          struct {
            int i;
            int j;
          } inner;
          int k;
        } o = {1, 2, 3};
      )cpp",
     false},
    {R"cpp(
        struct Outer {
          struct {
            int i;
            int j;
          } inner;
          int k;
        } o = {{1, 2}, 3};
      )cpp",
     true},
};

class NestedInitListTest : public testing::TestWithParam<NestedInitListParams> {
};

TEST_P(NestedInitListTest, IsExplicit) {
  auto AST = buildASTFromCode(GetParam().Code);
  const auto *OuterList = selectFirst<InitListExpr>(
      "initList", match(initListExpr().bind("initList"), AST->getASTContext()));
  ASSERT_NE(OuterList, nullptr);
  EXPECT_TRUE(OuterList->isExplicit());
  ASSERT_FALSE(OuterList->isSyntacticForm());
  EXPECT_TRUE(OuterList->getSyntacticForm()->isExplicit());
  ASSERT_TRUE(OuterList->getNumInits() >= 1);
  const auto *InnerList = dyn_cast<InitListExpr>(OuterList->getInit(0));
  ASSERT_NE(InnerList, nullptr);
  EXPECT_EQ(InnerList->isExplicit(), GetParam().InnerIsExplicit);
  if (!InnerList->isSyntacticForm()) {
    EXPECT_EQ(InnerList->getSyntacticForm()->isExplicit(),
              GetParam().InnerIsExplicit);
  }
}

INSTANTIATE_TEST_SUITE_P(NestedInitLists, NestedInitListTest,
                         testing::ValuesIn(NestedInitListParamArray));

TEST(ASTExpr, RewrappedExplicitInitList) {
  auto AST = buildASTFromCode(R"cpp(
    const int& x = {1};
  )cpp");
  const auto *InitList = selectFirst<InitListExpr>(
      "initList", match(initListExpr().bind("initList"), AST->getASTContext()));
  ASSERT_NE(InitList, nullptr);
  EXPECT_TRUE(InitList->isExplicit());
}

TEST(ASTExpr, DesignatedInitImplicitList) {
  auto AST = buildASTFromCodeWithArgs(R"c(
    struct Inner {
      int i, j;
    };
    struct Outer {
      struct Inner inner;
    } o = { .inner.i = 1 };
  )c",
                                      {"-xc"});
  const auto *OuterList = selectFirst<InitListExpr>(
      "initList", match(initListExpr().bind("initList"), AST->getASTContext()));
  ASSERT_NE(OuterList, nullptr);
  EXPECT_TRUE(OuterList->isExplicit());
  ASSERT_FALSE(OuterList->isSyntacticForm());
  EXPECT_TRUE(OuterList->getSyntacticForm()->isExplicit());
  ASSERT_TRUE(OuterList->getNumInits() == 1);
  const auto *InnerList = dyn_cast<InitListExpr>(OuterList->getInit(0));
  EXPECT_TRUE(InnerList->isSemanticForm() && InnerList->isSyntacticForm());
  EXPECT_FALSE(InnerList->isExplicit());
}

TEST(ASTExpr, DesignatedInitUpdateImplicitList) {
  auto AST = buildASTFromCodeWithArgs(R"c(
    struct Inner {
      int i, j;
    };
    struct Outer {
      struct Inner inner;
    } o = { (struct Inner){}, .inner.j = 1 };
  )c",
                                      {"-xc", "-Wno-initializer-overrides"});
  const auto *OuterList = selectFirst<InitListExpr>(
      "initList", match(initListExpr().bind("initList"), AST->getASTContext()));
  ASSERT_NE(OuterList, nullptr);
  ASSERT_EQ(OuterList->getNumInits(), 1u);
  const auto *UpdateExpr =
      dyn_cast<DesignatedInitUpdateExpr>(OuterList->getInit(0));
  const InitListExpr *UpdaterInit = UpdateExpr->getUpdater();
  EXPECT_FALSE(UpdaterInit->isExplicit());
}

TEST(ASTExpr, OpenCLVectorInitList) {
  auto AST = buildASTFromCodeWithArgs(R"c(
    void Fn() {
      (void)(int __attribute__((ext_vector_type(2))))(1,2);
    }
  )c",
                                      {"-xcl"});
  const auto *List = selectFirst<InitListExpr>(
      "initList", match(initListExpr().bind("initList"), AST->getASTContext()));
  EXPECT_FALSE(List->isExplicit());
}

TEST(ASTExpr, TransparentUnionInitList) {
  auto AST = buildASTFromCodeWithArgs(R"c(
    union U {
      int i;
    } __attribute__((transparent_union));
    void TakeUnion(union U);
    void Fn() {
      TakeUnion(1);
    }
  )c",
                                      {"-xc"});
  const auto *List = selectFirst<InitListExpr>(
      "initList", match(initListExpr().bind("initList"), AST->getASTContext()));
  EXPECT_FALSE(List->isExplicit());
}

TEST(ASTExpr, ComplexTplArgInitList) {
  auto AST = buildASTFromCodeWithArgs(R"cpp(
    template <double _Complex C>
    void TplFn() {
      (void)C;
    }
    void Fn() {
      TplFn<1.0 + 2.0i>();
    }
  )cpp",
                                      {"-std=c++20"});
  const auto *SNTTP = selectFirst<SubstNonTypeTemplateParmExpr>(
      "SNTTP", match(substNonTypeTemplateParmExpr().bind("SNTTP"),
                     AST->getASTContext()));
  const auto *List = dyn_cast<InitListExpr>(SNTTP->getReplacement());
  ASSERT_NE(List, nullptr);
  EXPECT_FALSE(List->isExplicit());
}

class OpenACCReductionInitListTest
    : public testing::TestWithParam<const char *> {};

TEST_P(OpenACCReductionInitListTest, ImplicitInit) {
  auto AST = buildASTFromCodeWithArgs(GetParam(), {"-fopenacc"});
  const auto *ACCConstruct = selectFirst<OpenACCConstructStmt>(
      "accConstruct",
      match(stmt(has(forStmt())).bind("accConstruct"), AST->getASTContext()));
  ASSERT_NE(ACCConstruct, nullptr);
  ArrayRef<const OpenACCClause *> Clauses = ACCConstruct->clauses();
  ASSERT_EQ(Clauses.size(), 1u);
  const auto *ReductionClause = dyn_cast<OpenACCReductionClause>(Clauses[0]);
  ASSERT_NE(ReductionClause, nullptr);
  ArrayRef<OpenACCReductionRecipe> Recipes = ReductionClause->getRecipes();
  ASSERT_EQ(Recipes.size(), 1u);
  const auto *List = dyn_cast<InitListExpr>(Recipes[0].AllocaDecl->getInit());
  ASSERT_NE(List, nullptr);
  EXPECT_FALSE(List->isExplicit());
}

INSTANTIATE_TEST_SUITE_P(OpenACCReductionInitLists,
                         OpenACCReductionInitListTest,
                         testing::Values(
                             R"cpp(
  void Fn() {
    int sum = 0;
    #pragma acc parallel loop reduction(+:sum)
    for (int i = 0; i < 5; ++i)
      sum += i;
  }
)cpp",
                             R"cpp(
  void Fn() {
    unsigned res = ~0u;
    #pragma acc parallel loop reduction(&:res)
    for (int i = 0; i < 5; ++i)
      res &= i;
  }
)cpp"));

TEST(ASTExpr, OpenACCFirstPrivateInitList) {
  auto AST = buildASTFromCodeWithArgs(R"cpp(
    void Fn() {
      int arr[2] = {};
      #pragma acc parallel loop firstprivate(arr)
      for (int i = 0; i < 5; ++i)
        (void)(arr[0] - arr[1]);
    }
  )cpp",
                                      {"-fopenacc"});
  const auto *ACCConstruct = selectFirst<OpenACCConstructStmt>(
      "accConstruct",
      match(stmt(has(forStmt())).bind("accConstruct"), AST->getASTContext()));
  ASSERT_NE(ACCConstruct, nullptr);
  ArrayRef<const OpenACCClause *> Clauses = ACCConstruct->clauses();
  ASSERT_EQ(Clauses.size(), 1u);
  const auto *FirstPrivate = dyn_cast<OpenACCFirstPrivateClause>(Clauses[0]);
  ASSERT_NE(FirstPrivate, nullptr);
  ArrayRef<OpenACCFirstPrivateRecipe> Recipes = FirstPrivate->getInitRecipes();
  ASSERT_EQ(Recipes.size(), 1u);
  const auto *List = dyn_cast<InitListExpr>(Recipes[0].AllocaDecl->getInit());
  ASSERT_NE(List, nullptr);
  EXPECT_FALSE(List->isExplicit());
}

TEST(ASTExpr, IsKnownToHaveBooleanValue) {
  auto AST = tooling::buildASTFromCodeWithArgs(
      R"c(
    struct S {
      int signed_bf1 : 1;
      unsigned unsigned_bf1 : 1;
      unsigned unsigned_bf2 : 2;
    };

    _Bool bool_value;
    int int_value;
    unsigned _BitInt(1) unsigned_bitint1;
    unsigned _BitInt(2) unsigned_bitint2;
    struct S s;

    void f(void) {
      int from_bool = bool_value;
      int from_int = int_value;
      int from_signed_bitfield1 = s.signed_bf1;
      int from_bitfield1 = s.unsigned_bf1;
      int from_bitfield2 = s.unsigned_bf2;
      int from_bitint1 = unsigned_bitint1;
      int from_bitint2 = unsigned_bitint2;
    }
  )c",
      {"-std=c23"}, "input.c");
  ASSERT_TRUE(AST);

  auto ExpectKnown = [&](const char *Name, bool Semantic, bool NonSemantic) {
    const VarDecl *VD = getVariableNode(AST.get(), Name);
    ASSERT_NE(VD, nullptr);
    ASSERT_TRUE(VD->hasInit());
    const Expr *Init = VD->getInit();
    EXPECT_EQ(Semantic, Init->isKnownToHaveBooleanValue(true)) << Name;
    EXPECT_EQ(NonSemantic, Init->isKnownToHaveBooleanValue(false)) << Name;
  };

  ExpectKnown("from_bool", true, true);
  ExpectKnown("from_int", false, false);
  ExpectKnown("from_signed_bitfield1", false, false);
  ExpectKnown("from_bitfield1", false, true);
  ExpectKnown("from_bitfield2", false, false);
  ExpectKnown("from_bitint1", false, true);
  ExpectKnown("from_bitint2", false, false);
}