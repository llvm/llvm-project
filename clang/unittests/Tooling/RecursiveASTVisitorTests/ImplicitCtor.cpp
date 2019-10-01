//===- unittest/Tooling/RecursiveASTVisitorTests/ImplicitCtor.cpp ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestVisitor.h"

using namespace clang;

namespace {

// A visitor that visits implicit declarations and matches constructors.
class ImplicitCtorVisitor
    : public ExpectedLocationVisitor<ImplicitCtorVisitor> {
public:
  bool shouldVisitImplicitCode() const { return true; }

  bool VisitCXXConstructorDecl(CXXConstructorDecl* Ctor) {
    if (Ctor->isImplicit()) {  // Was not written in source code
      if (const CXXRecordDecl* Class = Ctor->getParent()) {
        Match(Class->getName(), Ctor->getLocation());
      }
    }
    return true;
  }
};

TEST(RecursiveASTVisitor, VisitsImplicitCopyConstructors) {
  ImplicitCtorVisitor Visitor;
  Visitor.ExpectMatch("Simple", 2, 8);
  // Note: Clang lazily instantiates implicit declarations, so we need
  // to use them in order to force them to appear in the AST.
  EXPECT_TRUE(Visitor.runOver(
      "struct WithCtor { WithCtor(); }; \n"
      "struct Simple { Simple(); WithCtor w; }; \n"
      "int main() { Simple s; Simple t(s); }\n"));
}

} // end anonymous namespace
