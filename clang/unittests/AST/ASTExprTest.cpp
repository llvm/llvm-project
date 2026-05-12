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
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"

using namespace clang;

using clang::ast_matchers::cxxRecordDecl;
using clang::ast_matchers::hasName;
using clang::ast_matchers::match;
using clang::ast_matchers::varDecl;
using clang::tooling::buildASTFromCode;

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
  InitListExpr *BaseInit = new (Ctx) InitListExpr(Ctx, Loc, {}, Loc);
  BaseInit->setType(Ctx.getCanonicalTagType(Empty));
  Expr *Exprs[3] = {
      BaseInit,
      createIntLiteral(Ctx, 13),
      createIntLiteral(Ctx, 42),
  };
  InitListExpr *FooInit = new (Ctx) InitListExpr(Ctx, Loc, Exprs, Loc);
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
