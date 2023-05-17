//===- unittest/AST/ASTContextParentMapTest.cpp - AST parent map test -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for the getParents(...) methods of ASTContext.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "MatchVerifier.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

using testing::ElementsAre;

namespace clang {
namespace ast_matchers {

TEST(GetParents, ReturnsParentForDecl) {
  MatchVerifier<Decl> Verifier;
  EXPECT_TRUE(
      Verifier.match("class C { void f(); };",
                     cxxMethodDecl(hasParent(recordDecl(hasName("C"))))));
}

TEST(GetParents, ReturnsParentForStmt) {
  MatchVerifier<Stmt> Verifier;
  EXPECT_TRUE(Verifier.match("class C { void f() { if (true) {} } };",
                             ifStmt(hasParent(compoundStmt()))));
}

TEST(GetParents, ReturnsParentForTypeLoc) {
  MatchVerifier<TypeLoc> Verifier;
  EXPECT_TRUE(
      Verifier.match("namespace a { class b {}; } void f(a::b) {}",
                     typeLoc(hasParent(typeLoc(hasParent(functionDecl()))))));
}

TEST(GetParents, ReturnsParentForNestedNameSpecifierLoc) {
  MatchVerifier<NestedNameSpecifierLoc> Verifier;
  EXPECT_TRUE(Verifier.match("namespace a { class b {}; } void f(a::b) {}",
                             nestedNameSpecifierLoc(hasParent(typeLoc()))));
}

TEST(GetParents, ReturnsParentInsideTemplateInstantiations) {
  MatchVerifier<Decl> DeclVerifier;
  EXPECT_TRUE(DeclVerifier.match(
      "template<typename T> struct C { void f() {} };"
      "void g() { C<int> c; c.f(); }",
      cxxMethodDecl(hasName("f"),
                 hasParent(cxxRecordDecl(isTemplateInstantiation())))));
  EXPECT_TRUE(DeclVerifier.match(
      "template<typename T> struct C { void f() {} };"
      "void g() { C<int> c; c.f(); }",
      cxxMethodDecl(hasName("f"),
                 hasParent(cxxRecordDecl(unless(isTemplateInstantiation()))))));
  EXPECT_FALSE(DeclVerifier.match(
      "template<typename T> struct C { void f() {} };"
      "void g() { C<int> c; c.f(); }",
      cxxMethodDecl(
          hasName("f"),
          allOf(hasParent(cxxRecordDecl(unless(isTemplateInstantiation()))),
                hasParent(cxxRecordDecl(isTemplateInstantiation()))))));
}

TEST(GetParents, ReturnsMultipleParentsInTemplateInstantiations) {
  MatchVerifier<Stmt> TemplateVerifier;
  EXPECT_TRUE(TemplateVerifier.match(
      "template<typename T> struct C { void f() {} };"
      "void g() { C<int> c; c.f(); }",
      compoundStmt(allOf(
          hasAncestor(cxxRecordDecl(isTemplateInstantiation())),
          hasAncestor(cxxRecordDecl(unless(isTemplateInstantiation())))))));
}

TEST(GetParents, RespectsTraversalScope) {
  auto AST = tooling::buildASTFromCode(
      "struct foo { int bar; }; struct baz{};", "foo.cpp",
      std::make_shared<PCHContainerOperations>());
  auto &Ctx = AST->getASTContext();
  auto &TU = *Ctx.getTranslationUnitDecl();
  auto &Foo = *TU.lookup(&Ctx.Idents.get("foo")).front();
  auto &Bar = *cast<DeclContext>(Foo).lookup(&Ctx.Idents.get("bar")).front();
  auto &Baz = *TU.lookup(&Ctx.Idents.get("baz")).front();

  // Initially, scope is the whole TU.
  EXPECT_THAT(Ctx.getParents(Bar), ElementsAre(DynTypedNode::create(Foo)));
  EXPECT_THAT(Ctx.getParents(Foo), ElementsAre(DynTypedNode::create(TU)));
  EXPECT_THAT(Ctx.getParents(Baz), ElementsAre(DynTypedNode::create(TU)));

  // Restrict the scope, now some parents are gone.
  Ctx.setTraversalScope({&Foo});
  EXPECT_THAT(Ctx.getParents(Bar), ElementsAre(DynTypedNode::create(Foo)));
  EXPECT_THAT(Ctx.getParents(Foo), ElementsAre(DynTypedNode::create(TU)));
  EXPECT_THAT(Ctx.getParents(Baz), ElementsAre());

  // Reset the scope, we get back the original results.
  Ctx.setTraversalScope({&TU});
  EXPECT_THAT(Ctx.getParents(Bar), ElementsAre(DynTypedNode::create(Foo)));
  EXPECT_THAT(Ctx.getParents(Foo), ElementsAre(DynTypedNode::create(TU)));
  EXPECT_THAT(Ctx.getParents(Baz), ElementsAre(DynTypedNode::create(TU)));
}

TEST(GetParents, ImplicitLambdaNodes) {
  MatchVerifier<Decl> LambdaVerifier;
  EXPECT_TRUE(LambdaVerifier.match(
      "auto x = []{int y;};",
      varDecl(hasName("y"), hasAncestor(functionDecl(
                                hasOverloadedOperatorName("()"),
                                hasParent(cxxRecordDecl(
                                    isImplicit(), hasParent(lambdaExpr())))))),
      Lang_CXX11));
}

TEST(GetParents, FriendTypeLoc) {
  auto AST = tooling::buildASTFromCode("struct A { friend struct Fr; };"
                                       "struct B { friend struct Fr; };"
                                       "struct Fr;");
  auto &Ctx = AST->getASTContext();
  auto &TU = *Ctx.getTranslationUnitDecl();
  auto &A = *TU.lookup(&Ctx.Idents.get("A")).front();
  auto &B = *TU.lookup(&Ctx.Idents.get("B")).front();
  auto &FrA = *cast<FriendDecl>(*++(cast<CXXRecordDecl>(A).decls_begin()));
  auto &FrB = *cast<FriendDecl>(*++(cast<CXXRecordDecl>(B).decls_begin()));
  TypeLoc FrALoc = FrA.getFriendType()->getTypeLoc();
  TypeLoc FrBLoc = FrB.getFriendType()->getTypeLoc();
  TagDecl *FrATagDecl =
      FrALoc.getTypePtr()->getAs<ElaboratedType>()->getOwnedTagDecl();
  TagDecl *FrBTagDecl =
      FrBLoc.getTypePtr()->getAs<ElaboratedType>()->getOwnedTagDecl();

  EXPECT_THAT(Ctx.getParents(A), ElementsAre(DynTypedNode::create(TU)));
  EXPECT_THAT(Ctx.getParents(B), ElementsAre(DynTypedNode::create(TU)));
  EXPECT_THAT(Ctx.getParents(FrA), ElementsAre(DynTypedNode::create(A)));
  EXPECT_THAT(Ctx.getParents(FrB), ElementsAre(DynTypedNode::create(B)));
  EXPECT_THAT(Ctx.getParents(FrALoc), ElementsAre(DynTypedNode::create(FrA)));
  EXPECT_THAT(Ctx.getParents(FrBLoc), ElementsAre(DynTypedNode::create(FrB)));
  EXPECT_TRUE(FrATagDecl);
  EXPECT_FALSE(FrBTagDecl);
  EXPECT_THAT(Ctx.getParents(*FrATagDecl),
              ElementsAre(DynTypedNode::create(FrA)));
}

} // end namespace ast_matchers
} // end namespace clang
