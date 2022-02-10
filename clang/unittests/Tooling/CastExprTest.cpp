//===- unittest/Tooling/CastExprTest.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestVisitor.h"

using namespace clang;

namespace {

struct CastExprVisitor : TestVisitor<CastExprVisitor> {
  std::function<void(ExplicitCastExpr *)> OnExplicitCast;

  bool VisitExplicitCastExpr(ExplicitCastExpr *Expr) {
    if (OnExplicitCast)
      OnExplicitCast(Expr);
    return true;
  }
};

TEST(CastExprTest, GetSubExprAsWrittenThroughMaterializedTemporary) {
  CastExprVisitor Visitor;
  Visitor.OnExplicitCast = [](ExplicitCastExpr *Expr) {
    auto Sub = Expr->getSubExprAsWritten();
    EXPECT_TRUE(isa<DeclRefExpr>(Sub))
        << "Expected DeclRefExpr, but saw " << Sub->getStmtClassName();
  };
  Visitor.runOver("struct S1 {};\n"
                  "struct S2 { operator S1(); };\n"
                  "S1 f(S2 s) { return static_cast<S1>(s); }\n");
}

// Verify that getSubExprAsWritten looks through a ConstantExpr in a scenario
// like
//
//   CXXFunctionalCastExpr functional cast to struct S <ConstructorConversion>
//   `-ConstantExpr 'S'
//     |-value: Struct
//     `-CXXConstructExpr 'S' 'void (int)'
//       `-IntegerLiteral 'int' 0
TEST(CastExprTest, GetSubExprAsWrittenThroughConstantExpr) {
  CastExprVisitor Visitor;
  Visitor.OnExplicitCast = [](ExplicitCastExpr *Expr) {
    auto *Sub = Expr->getSubExprAsWritten();
    EXPECT_TRUE(isa<IntegerLiteral>(Sub))
        << "Expected IntegerLiteral, but saw " << Sub->getStmtClassName();
  };
  Visitor.runOver("struct S { consteval S(int) {} };\n"
                  "S f() { return S(0); }\n",
                  CastExprVisitor::Lang_CXX2a);
}

} // namespace
