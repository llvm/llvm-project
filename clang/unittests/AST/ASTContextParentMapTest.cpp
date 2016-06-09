//===- unittest/AST/ASTContextParentMapTest.cpp - AST parent map test -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

} // end namespace ast_matchers
} // end namespace clang
