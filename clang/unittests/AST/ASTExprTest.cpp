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

TEST(ASTExpr, IgnoreExprCallbackForwarded) {
  constexpr char Code[] = "";
  auto AST = tooling::buildASTFromCodeWithArgs(Code, /*Args=*/{"-std=c++20"});
  ASTContext &Ctx = AST->getASTContext();

  auto createIntLiteral = [&](uint32_t Value) -> IntegerLiteral * {
    const int numBits = 32;
    return IntegerLiteral::Create(Ctx, llvm::APInt(numBits, Value),
                                  Ctx.UnsignedIntTy, {});
  };

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
    auto *IntExpr = createIntLiteral(10);
    ParenExpr *PE =
        new (Ctx) ParenExpr(SourceLocation{}, SourceLocation{}, IntExpr);
    EXPECT_EQ(IntExpr, IgnoreExprNodes(PE, IgnoreParens{}));
  }

  {
    IgnoreParens CB{};
    auto *IntExpr = createIntLiteral(10);
    ParenExpr *PE =
        new (Ctx) ParenExpr(SourceLocation{}, SourceLocation{}, IntExpr);
    EXPECT_EQ(nullptr, IgnoreExprNodes(PE, CB));
  }
}
