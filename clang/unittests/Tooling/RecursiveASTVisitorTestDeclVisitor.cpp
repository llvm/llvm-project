//===- unittest/Tooling/RecursiveASTVisitorTestDeclVisitor.cpp ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestVisitor.h"

using namespace clang;

namespace {

class VarDeclVisitor : public ExpectedLocationVisitor<VarDeclVisitor> {
public:
 bool VisitVarDecl(VarDecl *Variable) {
   Match(Variable->getNameAsString(), Variable->getBeginLoc());
   return true;
 }
};

TEST(RecursiveASTVisitor, VisitsCXXForRangeStmtLoopVariable) {
  VarDeclVisitor Visitor;
  Visitor.ExpectMatch("i", 2, 17);
  EXPECT_TRUE(Visitor.runOver(
    "int x[5];\n"
    "void f() { for (int i : x) {} }",
    VarDeclVisitor::Lang_CXX11));
}

class ParmVarDeclVisitorForImplicitCode :
  public ExpectedLocationVisitor<ParmVarDeclVisitorForImplicitCode> {
public:
  bool shouldVisitImplicitCode() const { return true; }

  bool VisitParmVarDecl(ParmVarDecl *ParamVar) {
    Match(ParamVar->getNameAsString(), ParamVar->getBeginLoc());
    return true;
  }
};

// Test RAV visits parameter variable declaration of the implicit
// copy assignment operator and implicit copy constructor.
TEST(RecursiveASTVisitor, VisitsParmVarDeclForImplicitCode) {
  ParmVarDeclVisitorForImplicitCode Visitor;
  // Match parameter variable name of implicit copy assignment operator and
  // implicit copy constructor.
  // This parameter name does not have a valid IdentifierInfo, and shares
  // same SourceLocation with its class declaration, so we match an empty name
  // with the class' source location.
  Visitor.ExpectMatch("", 1, 7);
  Visitor.ExpectMatch("", 3, 7);
  EXPECT_TRUE(Visitor.runOver(
    "class X {};\n"
    "void foo(X a, X b) {a = b;}\n"
    "class Y {};\n"
    "void bar(Y a) {Y b = a;}"));
}

class NamedDeclVisitor
  : public ExpectedLocationVisitor<NamedDeclVisitor> {
public:
  bool VisitNamedDecl(NamedDecl *Decl) {
    std::string NameWithTemplateArgs;
    llvm::raw_string_ostream OS(NameWithTemplateArgs);
    Decl->getNameForDiagnostic(OS,
                               Decl->getASTContext().getPrintingPolicy(),
                               true);
    Match(NameWithTemplateArgs, Decl->getLocation());
    return true;
  }
};

TEST(RecursiveASTVisitor, VisitsPartialTemplateSpecialization) {
  // From cfe-commits/Week-of-Mon-20100830/033998.html
  // Contrary to the approach suggested in that email, we visit all
  // specializations when we visit the primary template.  Visiting them when we
  // visit the associated specialization is problematic for specializations of
  // template members of class templates.
  NamedDeclVisitor Visitor;
  Visitor.ExpectMatch("A<bool>", 1, 26);
  Visitor.ExpectMatch("A<char *>", 2, 26);
  EXPECT_TRUE(Visitor.runOver(
    "template <class T> class A {};\n"
    "template <class T> class A<T*> {};\n"
    "A<bool> ab;\n"
    "A<char*> acp;\n"));
}

TEST(RecursiveASTVisitor, VisitsUndefinedClassTemplateSpecialization) {
  NamedDeclVisitor Visitor;
  Visitor.ExpectMatch("A<int>", 1, 29);
  EXPECT_TRUE(Visitor.runOver(
    "template<typename T> struct A;\n"
    "A<int> *p;\n"));
}

TEST(RecursiveASTVisitor, VisitsNestedUndefinedClassTemplateSpecialization) {
  NamedDeclVisitor Visitor;
  Visitor.ExpectMatch("A<int>::B<char>", 2, 31);
  EXPECT_TRUE(Visitor.runOver(
    "template<typename T> struct A {\n"
    "  template<typename U> struct B;\n"
    "};\n"
    "A<int>::B<char> *p;\n"));
}

TEST(RecursiveASTVisitor, VisitsUndefinedFunctionTemplateSpecialization) {
  NamedDeclVisitor Visitor;
  Visitor.ExpectMatch("A<int>", 1, 26);
  EXPECT_TRUE(Visitor.runOver(
    "template<typename T> int A();\n"
    "int k = A<int>();\n"));
}

TEST(RecursiveASTVisitor, VisitsNestedUndefinedFunctionTemplateSpecialization) {
  NamedDeclVisitor Visitor;
  Visitor.ExpectMatch("A<int>::B<char>", 2, 35);
  EXPECT_TRUE(Visitor.runOver(
    "template<typename T> struct A {\n"
    "  template<typename U> static int B();\n"
    "};\n"
    "int k = A<int>::B<char>();\n"));
}

TEST(RecursiveASTVisitor, NoRecursionInSelfFriend) {
  // From cfe-commits/Week-of-Mon-20100830/033977.html
  NamedDeclVisitor Visitor;
  Visitor.ExpectMatch("vector_iterator<int>", 2, 7);
  EXPECT_TRUE(Visitor.runOver(
    "template<typename Container>\n"
    "class vector_iterator {\n"
    "    template <typename C> friend class vector_iterator;\n"
    "};\n"
    "vector_iterator<int> it_int;\n"));
}

} // end anonymous namespace
