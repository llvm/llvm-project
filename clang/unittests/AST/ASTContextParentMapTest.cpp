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
  bool FrAOwnsTag = FrALoc.getTypePtr()->getAs<TagType>()->isTagOwned();
  TagDecl *FrATagDecl =
      FrALoc.getTypePtr()->getAs<TagType>()->getOriginalDecl();
  bool FrBOwnsTag = FrBLoc.getTypePtr()->getAs<TagType>()->isTagOwned();

  EXPECT_THAT(Ctx.getParents(A), ElementsAre(DynTypedNode::create(TU)));
  EXPECT_THAT(Ctx.getParents(B), ElementsAre(DynTypedNode::create(TU)));
  EXPECT_THAT(Ctx.getParents(FrA), ElementsAre(DynTypedNode::create(A)));
  EXPECT_THAT(Ctx.getParents(FrB), ElementsAre(DynTypedNode::create(B)));
  EXPECT_THAT(Ctx.getParents(FrALoc), ElementsAre(DynTypedNode::create(FrA)));
  EXPECT_THAT(Ctx.getParents(FrBLoc), ElementsAre(DynTypedNode::create(FrB)));
  EXPECT_TRUE(FrAOwnsTag);
  EXPECT_FALSE(FrBOwnsTag);
  EXPECT_THAT(Ctx.getParents(*FrATagDecl),
              ElementsAre(DynTypedNode::create(FrA)));
}

TEST(GetParents, UserDefinedTupleLikeTypes) {
  MatchVerifier<VarDecl> Verifier;
  EXPECT_TRUE(Verifier.match(
      R"(
namespace std {

using size_t = __typeof(sizeof(int));

template <typename T>
struct tuple_size;

template <typename T>
struct tuple_size<T&> : tuple_size<T>{};

template <typename T>
requires requires { tuple_size<T>::value; }
struct tuple_size<const T> : tuple_size<T>{};


template<size_t i, typename T>
struct tuple_element;


}  // namespace std

struct Decomposable {};

template<> struct std::tuple_size<Decomposable> {
  static constexpr size_t value = 2;
};

template<std::size_t i> struct std::tuple_element<i, Decomposable> {
  using type = int;
};

template<std::size_t i> struct std::tuple_element<i, const Decomposable> {
  using type = const int;
};

template<std::size_t i>
const int& get(const Decomposable& d);

void F(const Decomposable& d) {
    const auto& [x, y] = d;
}
)",
      varDecl(hasName("x"), hasAncestor(decompositionDecl())), Lang_CXX20));
}

} // end namespace ast_matchers
} // end namespace clang
