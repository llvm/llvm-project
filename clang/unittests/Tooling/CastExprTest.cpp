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
  std::function<void(CastExpr *)> OnCast;

  bool VisitExplicitCastExpr(ExplicitCastExpr *Expr) {
    if (OnExplicitCast)
      OnExplicitCast(Expr);
    return true;
  }

  bool VisitCastExpr(CastExpr *Expr) {
    if (OnCast)
      OnCast(Expr);
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

// Verify that getConversionFunction looks through a ConstantExpr for implicit
// constructor conversions (https://github.com/llvm/llvm-project/issues/53044):
//
// `-ImplicitCastExpr 'X' <ConstructorConversion>
//   `-ConstantExpr 'X'
//     |-value: Struct
//     `-CXXConstructExpr 'X' 'void (const char *)'
//       `-ImplicitCastExpr 'const char *' <ArrayToPointerDecay>
//         `-StringLiteral 'const char [7]' lvalue "foobar"
TEST(CastExprTest, GetCtorConversionFunctionThroughConstantExpr) {
  CastExprVisitor Visitor;
  Visitor.OnCast = [](CastExpr *Expr) {
    if (Expr->getCastKind() == CK_ConstructorConversion) {
      auto *Conv = Expr->getConversionFunction();
      EXPECT_TRUE(isa<CXXConstructorDecl>(Conv))
          << "Expected CXXConstructorDecl, but saw " << Conv->getDeclKindName();
    }
  };
  Visitor.runOver("struct X { consteval X(const char *) {} };\n"
                  "void f() { X x = \"foobar\"; }\n",
                  CastExprVisitor::Lang_CXX2a);
}

// Verify that getConversionFunction looks through a ConstantExpr for implicit
// user-defined conversions.
//
// `-ImplicitCastExpr 'const char *' <UserDefinedConversion>
//   `-ConstantExpr 'const char *'
//     |-value: LValue
//     `-CXXMemberCallExpr 'const char *'
//       `-MemberExpr '<bound member function type>' .operator const char *
//         `-DeclRefExpr 'const X' lvalue Var 'x' 'const X'
TEST(CastExprTest, GetUserDefinedConversionFunctionThroughConstantExpr) {
  CastExprVisitor Visitor;
  Visitor.OnCast = [](CastExpr *Expr) {
    if (Expr->getCastKind() == CK_UserDefinedConversion) {
      auto *Conv = Expr->getConversionFunction();
      EXPECT_TRUE(isa<CXXMethodDecl>(Conv))
          << "Expected CXXMethodDecl, but saw " << Conv->getDeclKindName();
    }
  };
  Visitor.runOver("struct X {\n"
                  "  consteval operator const char *() const {\n"
                  "    return nullptr;\n"
                  "  }\n"
                  "};\n"
                  "const char *f() {\n"
                  "  constexpr X x;\n"
                  "  return x;\n"
                  "}\n",
                  CastExprVisitor::Lang_CXX2a);
}

} // namespace
