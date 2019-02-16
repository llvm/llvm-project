//===- unittest/Tooling/RecursiveASTVisitorTests/TraversalScope.cpp -------===//
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
