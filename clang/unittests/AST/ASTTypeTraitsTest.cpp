//===- unittest/AST/ASTTypeTraits.cpp - AST type traits unit tests ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------------===//


#include "clang/AST/ASTTypeTraits.h"
#include "MatchVerifier.h"
#include "gtest/gtest.h"

using namespace clang::ast_matchers;

namespace clang {
namespace {

TEST(ASTNodeKind, NoKind) {
  EXPECT_FALSE(ASTNodeKind().isBaseOf(ASTNodeKind()));
  EXPECT_FALSE(ASTNodeKind().isSame(ASTNodeKind()));
}

template <typename T> static ASTNodeKind DNT() {
  return ASTNodeKind::getFromNodeKind<T>();
}

TEST(ASTNodeKind, IsNone) {
  EXPECT_TRUE(ASTNodeKind().isNone());
  EXPECT_FALSE(DNT<Decl>().isNone());
  EXPECT_FALSE(DNT<VarDecl>().isNone());
}

TEST(ASTNodeKind, Bases) {
  EXPECT_TRUE(DNT<Decl>().isBaseOf(DNT<VarDecl>()));
  EXPECT_FALSE(DNT<Decl>().isSame(DNT<VarDecl>()));
  EXPECT_FALSE(DNT<VarDecl>().isBaseOf(DNT<Decl>()));

  EXPECT_TRUE(DNT<Decl>().isSame(DNT<Decl>()));
}

TEST(DynTypedNode, Clades) {
  EXPECT_TRUE(DNT<Stmt>().getCladeKind().isSame(DNT<Stmt>()));
  EXPECT_TRUE(DNT<Decl>().getCladeKind().isSame(DNT<Decl>()));

  EXPECT_TRUE(DNT<CXXMethodDecl>().getCladeKind().isSame(DNT<Decl>()));
  EXPECT_TRUE(DNT<CXXMemberCallExpr>().getCladeKind().isSame(DNT<Stmt>()));

  EXPECT_FALSE(DNT<CXXMemberCallExpr>().getCladeKind().isSame(DNT<Decl>()));

  EXPECT_TRUE(ASTNodeKind().getCladeKind().isNone());
}

TEST(ASTNodeKind, BaseDistances) {
  unsigned Distance = 1;
  EXPECT_TRUE(DNT<Expr>().isBaseOf(DNT<Expr>(), &Distance));
  EXPECT_EQ(0u, Distance);

  EXPECT_TRUE(DNT<Stmt>().isBaseOf(DNT<IfStmt>(), &Distance));
  EXPECT_EQ(1u, Distance);

  Distance = 3;
  EXPECT_TRUE(DNT<DeclaratorDecl>().isBaseOf(DNT<ParmVarDecl>(), &Distance));
  EXPECT_EQ(2u, Distance);
}

TEST(ASTNodeKind, SameBase) {
  EXPECT_TRUE(DNT<Expr>().isBaseOf(DNT<CallExpr>()));
  EXPECT_TRUE(DNT<Expr>().isBaseOf(DNT<BinaryOperator>()));
  EXPECT_FALSE(DNT<CallExpr>().isBaseOf(DNT<BinaryOperator>()));
  EXPECT_FALSE(DNT<BinaryOperator>().isBaseOf(DNT<CallExpr>()));
}

TEST(ASTNodeKind, DiffBase) {
  EXPECT_FALSE(DNT<Expr>().isBaseOf(DNT<ArrayType>()));
  EXPECT_FALSE(DNT<QualType>().isBaseOf(DNT<FunctionDecl>()));
  EXPECT_FALSE(DNT<Type>().isSame(DNT<QualType>()));
}

TEST(ASTNodeKind, MostDerivedType) {
  EXPECT_TRUE(DNT<BinaryOperator>().isSame(
      ASTNodeKind::getMostDerivedType(DNT<Expr>(), DNT<BinaryOperator>())));
  EXPECT_TRUE(DNT<BinaryOperator>().isSame(
      ASTNodeKind::getMostDerivedType(DNT<BinaryOperator>(), DNT<Expr>())));
  EXPECT_TRUE(DNT<VarDecl>().isSame(
      ASTNodeKind::getMostDerivedType(DNT<VarDecl>(), DNT<VarDecl>())));

  // Not related. Returns nothing.
  EXPECT_TRUE(
      ASTNodeKind::getMostDerivedType(DNT<IfStmt>(), DNT<VarDecl>()).isNone());
  EXPECT_TRUE(ASTNodeKind::getMostDerivedType(DNT<IfStmt>(),
                                              DNT<BinaryOperator>()).isNone());
}

TEST(ASTNodeKind, MostDerivedCommonAncestor) {
  EXPECT_TRUE(DNT<Expr>().isSame(ASTNodeKind::getMostDerivedCommonAncestor(
      DNT<Expr>(), DNT<BinaryOperator>())));
  EXPECT_TRUE(DNT<Expr>().isSame(ASTNodeKind::getMostDerivedCommonAncestor(
      DNT<BinaryOperator>(), DNT<Expr>())));
  EXPECT_TRUE(DNT<VarDecl>().isSame(ASTNodeKind::getMostDerivedCommonAncestor(
      DNT<VarDecl>(), DNT<VarDecl>())));

  // A little related. Returns the ancestor.
  EXPECT_TRUE(
      DNT<NamedDecl>().isSame(ASTNodeKind::getMostDerivedCommonAncestor(
          DNT<CXXMethodDecl>(), DNT<RecordDecl>())));

  // Not related. Returns nothing.
  EXPECT_TRUE(ASTNodeKind::getMostDerivedCommonAncestor(
                  DNT<IfStmt>(), DNT<VarDecl>()).isNone());
}

struct Foo {};

TEST(ASTNodeKind, UnknownKind) {
  // We can construct one, but it is nowhere in the hierarchy.
  EXPECT_FALSE(DNT<Foo>().isSame(DNT<Foo>()));
}

template <typename T>
constexpr bool HasPointerIdentity =
    ASTNodeKind::getFromNodeKind<T>().hasPointerIdentity();

TEST(ASTNodeKind, ConstexprHasPointerIdentity) {
  EXPECT_TRUE(HasPointerIdentity<Decl>);
  EXPECT_TRUE(HasPointerIdentity<Stmt>);
  EXPECT_FALSE(HasPointerIdentity<TypeLoc>);
  EXPECT_FALSE(HasPointerIdentity<QualType>);
  EXPECT_FALSE(HasPointerIdentity<Foo>);

  constexpr bool DefaultConstructedHasPointerIdentity =
      ASTNodeKind().hasPointerIdentity();
  EXPECT_FALSE(DefaultConstructedHasPointerIdentity);
}

template <typename T, typename U>
constexpr bool NodeKindIsSame =
    ASTNodeKind::getFromNodeKind<T>().isSame(ASTNodeKind::getFromNodeKind<U>());

TEST(ASTNodeKind, ConstexprIsSame) {
  EXPECT_TRUE((NodeKindIsSame<Decl, Decl>));
  EXPECT_FALSE((NodeKindIsSame<Decl, VarDecl>));
  EXPECT_FALSE((NodeKindIsSame<Foo, Foo>));

  constexpr bool DefaultConstructedIsSameToDefaultConstructed =
      ASTNodeKind().isSame(ASTNodeKind());
  EXPECT_FALSE(DefaultConstructedIsSameToDefaultConstructed);
}

template <typename T>
constexpr bool NodeKindIsNone = ASTNodeKind::getFromNodeKind<T>().isNone();

TEST(ASTNodeKind, ConstexprIsNone) {
  EXPECT_FALSE(NodeKindIsNone<Decl>);
  EXPECT_TRUE(NodeKindIsNone<Foo>);

  constexpr bool DefaultConstructedIsNone = ASTNodeKind().isNone();
  EXPECT_TRUE(DefaultConstructedIsNone);
}

TEST(ASTNodeKind, Name) {
  EXPECT_EQ("<None>", ASTNodeKind().asStringRef());
#define VERIFY_NAME(Node) EXPECT_EQ(#Node, DNT<Node>().asStringRef());
  VERIFY_NAME(TemplateArgument);
  VERIFY_NAME(NestedNameSpecifierLoc);
  VERIFY_NAME(QualType);
  VERIFY_NAME(TypeLoc);
  VERIFY_NAME(CXXCtorInitializer);
  VERIFY_NAME(ConceptReference);
  VERIFY_NAME(NestedNameSpecifier);
  VERIFY_NAME(Decl);
  VERIFY_NAME(CXXRecordDecl);
  VERIFY_NAME(Stmt);
  VERIFY_NAME(CallExpr);
  VERIFY_NAME(Type);
  VERIFY_NAME(ConstantArrayType);
  VERIFY_NAME(NonNullAttr);
#undef VERIFY_NAME
}

TEST(DynTypedNode, DeclSourceRange) {
  RangeVerifier<DynTypedNode> Verifier;
  Verifier.expectRange(1, 1, 1, 11);
  EXPECT_TRUE(Verifier.match("void f() {}", decl()));
}

TEST(DynTypedNode, StmtSourceRange) {
  RangeVerifier<DynTypedNode> Verifier;
  Verifier.expectRange(1, 10, 1, 11);
  EXPECT_TRUE(Verifier.match("void f() {}", stmt()));
}

TEST(DynTypedNode, TypeLocSourceRange) {
  RangeVerifier<DynTypedNode> Verifier;
  Verifier.expectRange(1, 1, 1, 8);
  EXPECT_TRUE(Verifier.match("void f() {}", typeLoc(loc(functionType()))));
}

TEST(DynTypedNode, NNSLocSourceRange) {
  RangeVerifier<DynTypedNode> Verifier;
  Verifier.expectRange(1, 33, 1, 34);
  EXPECT_TRUE(Verifier.match("namespace N { typedef void T; } N::T f() {}",
                             nestedNameSpecifierLoc()));
}

TEST(DynTypedNode, AttrSourceRange) {
  RangeVerifier<DynTypedNode> Verifier;
  Verifier.expectRange(1, 31, 1, 31);
  EXPECT_TRUE(Verifier.match("void x(char *y __attribute__((nonnull)) );",
                             ast_matchers::attr()));
}

// FIXME: add tests for ConceptReference once we add an ASTMatcher.

TEST(DynTypedNode, DeclDump) {
  DumpVerifier Verifier;
  Verifier.expectSubstring("FunctionDecl");
  EXPECT_TRUE(Verifier.match("void f() {}", functionDecl()));
}

TEST(DynTypedNode, StmtDump) {
  DumpVerifier Verifier;
  Verifier.expectSubstring("CompoundStmt");
  EXPECT_TRUE(Verifier.match("void f() {}", stmt()));
}

TEST(DynTypedNode, DeclPrint) {
  PrintVerifier Verifier;
  Verifier.expectString("void f() {\n}\n");
  EXPECT_TRUE(Verifier.match("void f() {}", functionDecl()));
}

TEST(DynTypedNode, StmtPrint) {
  PrintVerifier Verifier;
  Verifier.expectString("{\n}\n");
  EXPECT_TRUE(Verifier.match("void f() {}", stmt()));
}

TEST(DynTypedNode, QualType) {
  QualType Q;
  DynTypedNode Node = DynTypedNode::create(Q);
  EXPECT_TRUE(Node == Node);
  EXPECT_FALSE(Node < Node);
}

TEST(DynTypedNode, TypeLoc) {
  std::string code = R"cc(void example() { int abc; })cc";
  auto AST = clang::tooling::buildASTFromCode(code);
  auto matches =
      match(traverse(TK_AsIs,
                     varDecl(hasName("abc"), hasTypeLoc(typeLoc().bind("tl")))),
            AST->getASTContext());
  EXPECT_EQ(matches.size(), 1u);

  const auto &tl = *matches[0].getNodeAs<TypeLoc>("tl");
  DynTypedNode Node = DynTypedNode::create(tl);
  EXPECT_TRUE(Node == Node);
  EXPECT_FALSE(Node < Node);
}

TEST(DynTypedNode, PointerTypeLoc) {
  std::string code = R"cc(void example() { int *abc; })cc";
  auto AST = clang::tooling::buildASTFromCode(code);
  auto matches =
      match(traverse(TK_AsIs, varDecl(hasName("abc"),
                                      hasTypeLoc(typeLoc().bind("ptl")))),
            AST->getASTContext());
  EXPECT_EQ(matches.size(), 1u);

  const auto &tl = *matches[0].getNodeAs<TypeLoc>("ptl");
  DynTypedNode TypeLocNode = DynTypedNode::create(tl);
  EXPECT_TRUE(TypeLocNode == TypeLocNode);
  EXPECT_FALSE(TypeLocNode < TypeLocNode);

  const auto &ptl = *matches[0].getNodeAs<PointerTypeLoc>("ptl");
  EXPECT_EQ(&tl, &ptl);
  DynTypedNode PointerTypeLocNode = DynTypedNode::create(ptl);
  EXPECT_TRUE(PointerTypeLocNode == PointerTypeLocNode);
  EXPECT_FALSE(PointerTypeLocNode < PointerTypeLocNode);
}

} // namespace
}  // namespace clang
