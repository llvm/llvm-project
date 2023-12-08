//=------ unittest/Tooling/RecursiveASTVisitorTests/CXXMethodDecl.cpp ------=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestVisitor.h"
#include "clang/AST/Expr.h"

using namespace clang;

namespace {

class CXXMethodDeclVisitor
    : public ExpectedLocationVisitor<CXXMethodDeclVisitor> {
public:
  CXXMethodDeclVisitor(bool VisitImplicitCode)
      : VisitImplicitCode(VisitImplicitCode) {}

  bool shouldVisitImplicitCode() const { return VisitImplicitCode; }

  bool VisitDeclRefExpr(DeclRefExpr *D) {
    Match("declref", D->getLocation());
    return true;
  }
  bool VisitParmVarDecl(ParmVarDecl *P) {
    Match("parm", P->getLocation());
    return true;
  }

private:
  bool VisitImplicitCode;
};

TEST(RecursiveASTVisitor, CXXMethodDeclNoDefaultBodyVisited) {
  for (bool VisitImplCode : {false, true}) {
    CXXMethodDeclVisitor Visitor(VisitImplCode);
    if (VisitImplCode)
      Visitor.ExpectMatch("declref", 8, 28);
    else
      Visitor.DisallowMatch("declref", 8, 28);

    Visitor.ExpectMatch("parm", 8, 27);
    llvm::StringRef Code = R"cpp(
      struct B {};
      struct A {
        B BB;
        A &operator=(A &&O);
      };

      A &A::operator=(A &&O) = default;
    )cpp";
    EXPECT_TRUE(Visitor.runOver(Code, CXXMethodDeclVisitor::Lang_CXX11));
  }
}

TEST(RecursiveASTVisitor, FunctionDeclNoDefaultBodyVisited) {
  for (bool VisitImplCode : {false, true}) {
    CXXMethodDeclVisitor Visitor(VisitImplCode);
    if (VisitImplCode)
      Visitor.ExpectMatch("declref", 4, 58, /*Times=*/2);
    else
      Visitor.DisallowMatch("declref", 4, 58);
    llvm::StringRef Code = R"cpp(
      struct s {
        int x;
        friend auto operator==(s a, s b) -> bool = default;
      };
      bool k = s() == s(); // make sure clang generates the "==" definition.
    )cpp";
    EXPECT_TRUE(Visitor.runOver(Code, CXXMethodDeclVisitor::Lang_CXX2a));
  }
}
} // end anonymous namespace
