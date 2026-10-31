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
#include "clang/AST/RecursiveASTVisitor.h"
#include "MatchVerifier.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include <vector>

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

namespace {

/// Collects every LambdaExpr in the AST, including template instantiations.
class LambdaCollector : public RecursiveASTVisitor<LambdaCollector> {
public:
  bool shouldVisitTemplateInstantiations() const { return true; }

  bool VisitLambdaExpr(LambdaExpr *LE) {
    Lambdas.push_back(LE);
    return true;
  }

  std::vector<LambdaExpr *> Lambdas;
};

void collectDeclRefExprs(Stmt *S, std::vector<DeclRefExpr *> &Refs) {
  if (!S)
    return;
  if (auto *DRE = dyn_cast<DeclRefExpr>(S))
    Refs.push_back(DRE);
  for (Stmt *Child : S->children())
    collectDeclRefExprs(Child, Refs);
}

} // namespace

TEST(GetParents, MergedLambdaClosureBody) {
  // Two module units instantiate the same variable template specialization
  // containing a lambda. The closure types are merged and the call operator is
  // left without a body; see issue #236863.
  llvm::SmallString<256> ModuleCache;
  ASSERT_FALSE(llvm::sys::fs::createUniqueDirectory(
      "clang-parent-map-lambda-modules", ModuleCache));
  llvm::scope_exit CleanupModuleCache(
      [&] { llvm::sys::fs::remove_directories(ModuleCache); });

  auto AST = tooling::buildASTFromCodeWithArgs(
      R"cpp(
#pragma clang module build A
module A {}
#pragma clang module contents
#pragma clang module begin A
struct SharedType { int a{}; };
template <class T> inline constexpr unsigned long self = sizeof(T);
template <class T> inline constexpr auto probe = [] { return self<T>; }();
inline constexpr unsigned long kA = probe<SharedType>;
#pragma clang module end
#pragma clang module endbuild

#pragma clang module build B
module B {}
#pragma clang module contents
#pragma clang module begin B
struct SharedType { int a{}; };
template <class T> inline constexpr unsigned long self = sizeof(T);
template <class T> inline constexpr auto probe = [] { return self<T>; }();
inline constexpr unsigned long kB = probe<SharedType>;
#pragma clang module end
#pragma clang module endbuild

#pragma clang module import A
#pragma clang module import B

unsigned long use() { return kA + kB; }
)cpp",
      {"-std=c++20", "-fmodules",
       "-fmodules-cache-path=" + ModuleCache.str().str()});
  ASSERT_TRUE(AST);

  auto &Ctx = AST->getASTContext();
  // Build the parent map before anything below loads the call operator bodies:
  // deserializing the bodies changes which traversal path reaches them.
  (void)Ctx.getParents(*Ctx.getTranslationUnitDecl());

  LambdaCollector Collector;
  Collector.TraverseDecl(Ctx.getTranslationUnitDecl());

  unsigned NumBodies = 0;
  for (LambdaExpr *LE : Collector.Lambdas) {
    if (!LE->getLambdaClass()->isFromASTFile())
      continue;
    Stmt *Body = LE->getBody();
    ASSERT_NE(Body, nullptr);
    ++NumBodies;

    // The body is reached through the call operator, as for any other lambda.
    DynTypedNodeList BodyParents = Ctx.getParents(*Body);
    ASSERT_FALSE(BodyParents.empty());
    EXPECT_NE(BodyParents[0].get<CXXMethodDecl>(), nullptr);

    // These nodes' missing parents caused the clang-tidy crash.
    std::vector<DeclRefExpr *> Refs;
    collectDeclRefExprs(Body, Refs);
    for (DeclRefExpr *Ref : Refs)
      EXPECT_FALSE(Ctx.getParents(*Ref).empty());
  }
  EXPECT_GT(NumBodies, 0u);
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
  TagDecl *FrATagDecl = FrALoc.getTypePtr()->getAs<TagType>()->getDecl();
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
