//===- unittest/Tooling/RecursiveASTVisitorTests/TraversalScope.cpp -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestVisitor.h"

using namespace clang;

namespace {

class Visitor : public ExpectedLocationVisitor<Visitor, clang::TestVisitor> {
public:
  Visitor(ASTContext *Context) { this->Context = Context; }

  bool VisitNamedDecl(NamedDecl *D) {
    if (!D->isImplicit())
      Match(D->getName(), D->getLocation());
    return true;
  }
};

TEST(RecursiveASTVisitor, RespectsTraversalScope) {
  auto AST = tooling::buildASTFromCode(
      R"cpp(
struct foo {
  struct bar {
    struct baz {};
  };
};
      )cpp",
      "foo.cpp", std::make_shared<PCHContainerOperations>());
  auto &Ctx = AST->getASTContext();
  auto &TU = *Ctx.getTranslationUnitDecl();
  auto &Foo = *TU.lookup(&Ctx.Idents.get("foo")).front();
  auto &Bar = *cast<DeclContext>(Foo).lookup(&Ctx.Idents.get("bar")).front();

  Ctx.setTraversalScope({&Bar});

  Visitor V(&Ctx);
  V.DisallowMatch("foo", 2, 8);
  V.ExpectMatch("bar", 3, 10);
  V.ExpectMatch("baz", 4, 12);
  V.TraverseAST(Ctx);
}

} // end anonymous namespace
