//===- unittest/Tooling/CastExprTest.cpp ----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

}
