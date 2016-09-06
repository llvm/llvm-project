//===- unittest/Tooling/RecursiveASTVisitorTestCallVisitor.cpp ------------===//
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

class CXXMemberCallVisitor
  : public ExpectedLocationVisitor<CXXMemberCallVisitor> {
public:
  bool VisitCXXMemberCallExpr(CXXMemberCallExpr *Call) {
    Match(Call->getMethodDecl()->getQualifiedNameAsString(),
          Call->getLocStart());
    return true;
  }
};

TEST(RecursiveASTVisitor, VisitsCallInTemplateInstantiation) {
  CXXMemberCallVisitor Visitor;
  Visitor.ExpectMatch("Y::x", 3, 3);
  EXPECT_TRUE(Visitor.runOver(
    "struct Y { void x(); };\n"
    "template<typename T> void y(T t) {\n"
    "  t.x();\n"
    "}\n"
    "void foo() { y<Y>(Y()); }"));
}

TEST(RecursiveASTVisitor, VisitsCallInNestedFunctionTemplateInstantiation) {
  CXXMemberCallVisitor Visitor;
  Visitor.ExpectMatch("Y::x", 4, 5);
  EXPECT_TRUE(Visitor.runOver(
    "struct Y { void x(); };\n"
    "template<typename T> struct Z {\n"
    "  template<typename U> static void f() {\n"
    "    T().x();\n"
    "  }\n"
    "};\n"
    "void foo() { Z<Y>::f<int>(); }"));
}

TEST(RecursiveASTVisitor, VisitsCallInNestedClassTemplateInstantiation) {
  CXXMemberCallVisitor Visitor;
  Visitor.ExpectMatch("A::x", 5, 7);
  EXPECT_TRUE(Visitor.runOver(
    "template <typename T1> struct X {\n"
    "  template <typename T2> struct Y {\n"
    "    void f() {\n"
    "      T2 y;\n"
    "      y.x();\n"
    "    }\n"
    "  };\n"
    "};\n"
    "struct A { void x(); };\n"
    "int main() {\n"
    "  (new X<A>::Y<A>())->f();\n"
    "}"));
}

TEST(RecursiveASTVisitor, VisitsCallInPartialTemplateSpecialization) {
  CXXMemberCallVisitor Visitor;
  Visitor.ExpectMatch("A::x", 6, 20);
  EXPECT_TRUE(Visitor.runOver(
    "template <typename T1> struct X {\n"
    "  template <typename T2, bool B> struct Y { void g(); };\n"
    "};\n"
    "template <typename T1> template <typename T2>\n"
    "struct X<T1>::Y<T2, true> {\n"
    "  void f() { T2 y; y.x(); }\n"
    "};\n"
    "struct A { void x(); };\n"
    "int main() {\n"
    "  (new X<A>::Y<A, true>())->f();\n"
    "}\n"));
}

TEST(RecursiveASTVisitor, VisitsExplicitTemplateSpecialization) {
  CXXMemberCallVisitor Visitor;
  Visitor.ExpectMatch("A::f", 4, 5);
  EXPECT_TRUE(Visitor.runOver(
    "struct A {\n"
    "  void f() const {}\n"
    "  template<class T> void g(const T& t) const {\n"
    "    t.f();\n"
    "  }\n"
    "};\n"
    "template void A::g(const A& a) const;\n"));
}

class CXXOperatorCallExprTraverser
  : public ExpectedLocationVisitor<CXXOperatorCallExprTraverser> {
public:
  // Use Traverse, not Visit, to check that data recursion optimization isn't
  // bypassing the call of this function.
  bool TraverseCXXOperatorCallExpr(CXXOperatorCallExpr *CE) {
    Match(getOperatorSpelling(CE->getOperator()), CE->getExprLoc());
    return ExpectedLocationVisitor<CXXOperatorCallExprTraverser>::
        TraverseCXXOperatorCallExpr(CE);
  }
};

TEST(RecursiveASTVisitor, TraversesOverloadedOperator) {
  CXXOperatorCallExprTraverser Visitor;
  Visitor.ExpectMatch("()", 4, 9);
  EXPECT_TRUE(Visitor.runOver(
    "struct A {\n"
    "  int operator()();\n"
    "} a;\n"
    "int k = a();\n"));
}

} // end anonymous namespace
