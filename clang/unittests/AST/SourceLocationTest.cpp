//===- unittest/AST/SourceLocationTest.cpp - AST source loc unit tests ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains tests for SourceLocation and SourceRange fields
// in AST nodes.
//
// FIXME: In the long-term, when we test more than source locations, we may
// want to have a unit test file for an AST node (or group of related nodes),
// rather than a unit test file for source locations for all AST nodes.
//
//===----------------------------------------------------------------------===//

#include "MatchVerifier.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Testing/Support/Annotations.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace clang::ast_matchers;

namespace {

// FIXME: Pull the *Verifier tests into their own test file.

TEST(MatchVerifier, ParseError) {
  LocationVerifier<VarDecl> Verifier;
  Verifier.expectLocation(1, 1);
  EXPECT_FALSE(Verifier.match("int i", varDecl()));
}

TEST(MatchVerifier, NoMatch) {
  LocationVerifier<VarDecl> Verifier;
  Verifier.expectLocation(1, 1);
  EXPECT_FALSE(Verifier.match("int i;", recordDecl()));
}

TEST(MatchVerifier, WrongType) {
  LocationVerifier<RecordDecl> Verifier;
  Verifier.expectLocation(1, 1);
  EXPECT_FALSE(Verifier.match("int i;", varDecl()));
}

TEST(LocationVerifier, WrongLocation) {
  LocationVerifier<VarDecl> Verifier;
  Verifier.expectLocation(1, 1);
  EXPECT_FALSE(Verifier.match("int i;", varDecl()));
}

TEST(RangeVerifier, WrongRange) {
  RangeVerifier<VarDecl> Verifier;
  Verifier.expectRange(1, 1, 1, 1);
  EXPECT_FALSE(Verifier.match("int i;", varDecl()));
}

class WhileParenLocationVerifier : public MatchVerifier<WhileStmt> {
  unsigned ExpectLParenLine = 0, ExpectLParenColumn = 0;
  unsigned ExpectRParenLine = 0, ExpectRParenColumn = 0;

public:
  void expectLocations(unsigned LParenLine, unsigned LParenColumn,
                       unsigned RParenLine, unsigned RParenColumn) {
    ExpectLParenLine = LParenLine;
    ExpectLParenColumn = LParenColumn;
    ExpectRParenLine = RParenLine;
    ExpectRParenColumn = RParenColumn;
  }

protected:
  void verify(const MatchFinder::MatchResult &Result,
              const WhileStmt &Node) override {
    SourceLocation LParenLoc = Node.getLParenLoc();
    SourceLocation RParenLoc = Node.getRParenLoc();
    unsigned LParenLine =
        Result.SourceManager->getSpellingLineNumber(LParenLoc);
    unsigned LParenColumn =
        Result.SourceManager->getSpellingColumnNumber(LParenLoc);
    unsigned RParenLine =
        Result.SourceManager->getSpellingLineNumber(RParenLoc);
    unsigned RParenColumn =
        Result.SourceManager->getSpellingColumnNumber(RParenLoc);

    if (LParenLine != ExpectLParenLine || LParenColumn != ExpectLParenColumn ||
        RParenLine != ExpectRParenLine || RParenColumn != ExpectRParenColumn) {
      std::string MsgStr;
      llvm::raw_string_ostream Msg(MsgStr);
      Msg << "Expected LParen Location <" << ExpectLParenLine << ":"
          << ExpectLParenColumn << ">, found <";
      LParenLoc.print(Msg, *Result.SourceManager);
      Msg << ">\n";

      Msg << "Expected RParen Location <" << ExpectRParenLine << ":"
          << ExpectRParenColumn << ">, found <";
      RParenLoc.print(Msg, *Result.SourceManager);
      Msg << ">";

      this->setFailure(Msg.str());
    }
  }
};

TEST(LocationVerifier, WhileParenLoc) {
  WhileParenLocationVerifier Verifier;
  Verifier.expectLocations(1, 17, 1, 38);
  EXPECT_TRUE(Verifier.match("void f() { while(true/*some comment*/) {} }",
                             whileStmt()));
}

class LabelDeclRangeVerifier : public RangeVerifier<LabelStmt> {
protected:
  SourceRange getRange(const LabelStmt &Node) override {
    return Node.getDecl()->getSourceRange();
  }
};

TEST(LabelDecl, Range) {
  LabelDeclRangeVerifier Verifier;
  Verifier.expectRange(1, 12, 1, 12);
  EXPECT_TRUE(Verifier.match("void f() { l: return; }", labelStmt()));
}

TEST(LabelStmt, Range) {
  RangeVerifier<LabelStmt> Verifier;
  Verifier.expectRange(1, 12, 1, 15);
  EXPECT_TRUE(Verifier.match("void f() { l: return; }", labelStmt()));
}

TEST(ParmVarDecl, KNRLocation) {
  LocationVerifier<ParmVarDecl> Verifier;
  Verifier.expectLocation(1, 8);
  EXPECT_TRUE(Verifier.match("void f(i) {}", varDecl(), Lang_C99));
}

TEST(ParmVarDecl, KNRRange) {
  RangeVerifier<ParmVarDecl> Verifier;
  Verifier.expectRange(1, 8, 1, 8);
  EXPECT_TRUE(Verifier.match("void f(i) {}", varDecl(), Lang_C99));
}

TEST(CXXNewExpr, ArrayRange) {
  RangeVerifier<CXXNewExpr> Verifier;
  Verifier.expectRange(1, 12, 1, 22);
  EXPECT_TRUE(Verifier.match("void f() { new int[10]; }", cxxNewExpr()));
}

TEST(CXXNewExpr, ParenRange) {
  RangeVerifier<CXXNewExpr> Verifier;
  Verifier.expectRange(1, 12, 1, 20);
  EXPECT_TRUE(Verifier.match("void f() { new int(); }", cxxNewExpr()));
}

TEST(MemberExpr, ImplicitMemberRange) {
  RangeVerifier<MemberExpr> Verifier;
  Verifier.expectRange(2, 30, 2, 30);
  EXPECT_TRUE(Verifier.match("struct S { operator int() const; };\n"
                             "int foo(const S& s) { return s; }",
                             memberExpr()));
}

class MemberExprArrowLocVerifier : public RangeVerifier<MemberExpr> {
protected:
  SourceRange getRange(const MemberExpr &Node) override {
     return Node.getOperatorLoc();
  }
};

TEST(MemberExpr, ArrowRange) {
  MemberExprArrowLocVerifier Verifier;
  Verifier.expectRange(2, 19, 2, 19);
  EXPECT_TRUE(Verifier.match("struct S { int x; };\n"
                             "void foo(S *s) { s->x = 0; }",
                             memberExpr()));
}

TEST(MemberExpr, MacroArrowRange) {
  MemberExprArrowLocVerifier Verifier;
  Verifier.expectRange(1, 24, 1, 24);
  EXPECT_TRUE(Verifier.match("#define MEMBER(a, b) (a->b)\n"
                             "struct S { int x; };\n"
                             "void foo(S *s) { MEMBER(s, x) = 0; }",
                             memberExpr()));
}

TEST(MemberExpr, ImplicitArrowRange) {
  MemberExprArrowLocVerifier Verifier;
  Verifier.expectRange(0, 0, 0, 0);
  EXPECT_TRUE(Verifier.match("struct S { int x; void Test(); };\n"
                             "void S::Test() { x = 1; }",
                             memberExpr()));
}

TEST(VarDecl, VMTypeFixedVarDeclRange) {
  RangeVerifier<VarDecl> Verifier;
  Verifier.expectRange(1, 1, 1, 23);
  EXPECT_TRUE(Verifier.match("int a[(int)(void*)1234];",
                             varDecl(), Lang_C89));
}

TEST(TypeLoc, IntRange) {
  RangeVerifier<TypeLoc> Verifier;
  Verifier.expectRange(1, 1, 1, 1);
  EXPECT_TRUE(Verifier.match("int a;", typeLoc()));
}

TEST(TypeLoc, LongRange) {
  RangeVerifier<TypeLoc> Verifier;
  Verifier.expectRange(1, 1, 1, 1);
  EXPECT_TRUE(Verifier.match("long a;", typeLoc()));
}

TEST(TypeLoc, LongDoubleRange) {
  RangeVerifier<TypeLoc> Verifier;
  Verifier.expectRange(1, 1, 1, 6);
  EXPECT_TRUE(Verifier.match("long double a;", typeLoc()));
}

TEST(TypeLoc, DoubleLongRange) {
  RangeVerifier<TypeLoc> Verifier;
  Verifier.expectRange(1, 1, 1, 8);
  EXPECT_TRUE(Verifier.match("double long a;", typeLoc()));
}

TEST(TypeLoc, LongIntRange) {
  RangeVerifier<TypeLoc> Verifier;
  Verifier.expectRange(1, 1, 1, 6);
  EXPECT_TRUE(Verifier.match("long int a;", typeLoc()));
}

TEST(TypeLoc, IntLongRange) {
  RangeVerifier<TypeLoc> Verifier;
  Verifier.expectRange(1, 1, 1, 5);
  EXPECT_TRUE(Verifier.match("int long a;", typeLoc()));
}

TEST(TypeLoc, UnsignedIntRange) {
  RangeVerifier<TypeLoc> Verifier;
  Verifier.expectRange(1, 1, 1, 10);
  EXPECT_TRUE(Verifier.match("unsigned int a;", typeLoc()));
}

TEST(TypeLoc, IntUnsignedRange) {
  RangeVerifier<TypeLoc> Verifier;
  Verifier.expectRange(1, 1, 1, 5);
  EXPECT_TRUE(Verifier.match("int unsigned a;", typeLoc()));
}

TEST(TypeLoc, LongLongRange) {
  RangeVerifier<TypeLoc> Verifier;
  Verifier.expectRange(1, 1, 1, 6);
  EXPECT_TRUE(Verifier.match("long long a;", typeLoc()));
}

TEST(TypeLoc, UnsignedLongLongRange) {
  RangeVerifier<TypeLoc> Verifier;
  Verifier.expectRange(1, 1, 1, 15);
  EXPECT_TRUE(Verifier.match("unsigned long long a;", typeLoc()));
}

TEST(TypeLoc, LongUnsignedLongRange) {
  RangeVerifier<TypeLoc> Verifier;
  Verifier.expectRange(1, 1, 1, 15);
  EXPECT_TRUE(Verifier.match("long unsigned long a;", typeLoc()));
}

TEST(TypeLoc, LongLongUnsignedRange) {
  RangeVerifier<TypeLoc> Verifier;
  Verifier.expectRange(1, 1, 1, 11);
  EXPECT_TRUE(Verifier.match("long long unsigned a;", typeLoc()));
}

TEST(TypeLoc, ConstLongLongRange) {
  RangeVerifier<TypeLoc> Verifier;
  Verifier.expectRange(1, 7, 1, 12);
  EXPECT_TRUE(Verifier.match("const long long a = 0;", typeLoc()));
}

TEST(TypeLoc, LongConstLongRange) {
  RangeVerifier<TypeLoc> Verifier;
  Verifier.expectRange(1, 1, 1, 12);
  EXPECT_TRUE(Verifier.match("long const long a = 0;", typeLoc()));
}

TEST(TypeLoc, LongLongConstRange) {
  RangeVerifier<TypeLoc> Verifier;
  Verifier.expectRange(1, 1, 1, 6);
  EXPECT_TRUE(Verifier.match("long long const a = 0;", typeLoc()));
}

TEST(CXXConstructorDecl, NoRetFunTypeLocRange) {
  RangeVerifier<CXXConstructorDecl> Verifier;
  Verifier.expectRange(1, 11, 1, 13);
  EXPECT_TRUE(Verifier.match("class C { C(); };", functionDecl()));
}

TEST(CXXConstructorDecl, DefaultedCtorLocRange) {
  RangeVerifier<CXXConstructorDecl> Verifier;
  Verifier.expectRange(1, 11, 1, 23);
  EXPECT_TRUE(Verifier.match("class C { C() = default; };", functionDecl()));
}

TEST(CXXConstructorDecl, DeletedCtorLocRange) {
  RangeVerifier<CXXConstructorDecl> Verifier;
  Verifier.expectRange(1, 11, 1, 22);
  EXPECT_TRUE(Verifier.match("class C { C() = delete; };", functionDecl()));
}

TEST(CompoundLiteralExpr, CompoundVectorLiteralRange) {
  RangeVerifier<CompoundLiteralExpr> Verifier;
  Verifier.expectRange(2, 11, 2, 22);
  EXPECT_TRUE(Verifier.match(
                  "typedef int int2 __attribute__((ext_vector_type(2)));\n"
                  "int2 i2 = (int2){1, 2};", compoundLiteralExpr()));
}

TEST(CompoundLiteralExpr, ParensCompoundVectorLiteralRange) {
  RangeVerifier<CompoundLiteralExpr> Verifier;
  Verifier.expectRange(2, 20, 2, 31);
  EXPECT_TRUE(Verifier.match(
                  "typedef int int2 __attribute__((ext_vector_type(2)));\n"
                  "constant int2 i2 = (int2)(1, 2);", 
                  compoundLiteralExpr(), Lang_OpenCL));
}

TEST(InitListExpr, VectorLiteralListBraceRange) {
  RangeVerifier<InitListExpr> Verifier;
  Verifier.expectRange(2, 17, 2, 22);
  EXPECT_TRUE(Verifier.match(
                  "typedef int int2 __attribute__((ext_vector_type(2)));\n"
                  "int2 i2 = (int2){1, 2};", initListExpr()));
}

TEST(InitListExpr, VectorLiteralInitListParens) {
  RangeVerifier<InitListExpr> Verifier;
  Verifier.expectRange(2, 26, 2, 31);
  EXPECT_TRUE(Verifier.match(
                  "typedef int int2 __attribute__((ext_vector_type(2)));\n"
                  "constant int2 i2 = (int2)(1, 2);", initListExpr(), Lang_OpenCL));
}

class TemplateAngleBracketLocRangeVerifier : public RangeVerifier<TypeLoc> {
protected:
  SourceRange getRange(const TypeLoc &Node) override {
    TemplateSpecializationTypeLoc T =
        Node.getUnqualifiedLoc().castAs<TemplateSpecializationTypeLoc>();
    assert(!T.isNull());
    return SourceRange(T.getLAngleLoc(), T.getRAngleLoc());
  }
};

TEST(TemplateSpecializationTypeLoc, AngleBracketLocations) {
  TemplateAngleBracketLocRangeVerifier Verifier;
  Verifier.expectRange(2, 8, 2, 10);
  EXPECT_TRUE(Verifier.match(
      "template<typename T> struct A {}; struct B{}; void f(\n"
      "const A<B>&);",
      loc(templateSpecializationType())));
}

TEST(CXXNewExpr, TypeParenRange) {
  RangeVerifier<CXXNewExpr> Verifier;
  Verifier.expectRange(1, 10, 1, 18);
  EXPECT_TRUE(Verifier.match("int* a = new (int);", cxxNewExpr()));
}

class UnaryTransformTypeLocParensRangeVerifier : public RangeVerifier<TypeLoc> {
protected:
  SourceRange getRange(const TypeLoc &Node) override {
    UnaryTransformTypeLoc T =
        Node.getUnqualifiedLoc().castAs<UnaryTransformTypeLoc>();
    assert(!T.isNull());
    return SourceRange(T.getLParenLoc(), T.getRParenLoc());
  }
};

TEST(UnaryTransformTypeLoc, ParensRange) {
  UnaryTransformTypeLocParensRangeVerifier Verifier;
  Verifier.expectRange(3, 26, 3, 28);
  EXPECT_TRUE(Verifier.match(
      "template <typename T>\n"
      "struct S {\n"
      "typedef __underlying_type(T) type;\n"
      "};",
      loc(unaryTransformType())));
}

TEST(CXXFunctionalCastExpr, SourceRange) {
  RangeVerifier<CXXFunctionalCastExpr> Verifier;
  Verifier.expectRange(2, 10, 2, 14);
  EXPECT_TRUE(Verifier.match(
      "int foo() {\n"
      "  return int{};\n"
      "}",
      cxxFunctionalCastExpr(), Lang_CXX11));
}

TEST(CXXConstructExpr, SourceRange) {
  RangeVerifier<CXXConstructExpr> Verifier;
  Verifier.expectRange(3, 14, 3, 19);
  EXPECT_TRUE(Verifier.match(
      "struct A { A(int, int); };\n"
      "void f(A a);\n"
      "void g() { f({0, 0}); }",
      cxxConstructExpr(), Lang_CXX11));
}

TEST(CXXTemporaryObjectExpr, SourceRange) {
  RangeVerifier<CXXTemporaryObjectExpr> Verifier;
  Verifier.expectRange(2, 6, 2, 12);
  EXPECT_TRUE(Verifier.match(
      "struct A { A(int, int); };\n"
      "A a( A{0, 0} );",
      cxxTemporaryObjectExpr(), Lang_CXX11));
}

TEST(CXXUnresolvedConstructExpr, SourceRange) {
  RangeVerifier<CXXUnresolvedConstructExpr> Verifier;
  Verifier.expectRange(3, 10, 3, 12);
  std::vector<std::string> Args;
  Args.push_back("-fno-delayed-template-parsing");
  EXPECT_TRUE(Verifier.match(
      "template <typename U>\n"
      "U foo() {\n"
      "  return U{};\n"
      "}",
      cxxUnresolvedConstructExpr(), Args, Lang_CXX11));
}

TEST(UsingDecl, SourceRange) {
  RangeVerifier<UsingDecl> Verifier;
  Verifier.expectRange(2, 22, 2, 25);
  EXPECT_TRUE(Verifier.match(
      "class B { protected: int i; };\n"
      "class D : public B { B::i; };",
      usingDecl()));
}

TEST(UnresolvedUsingValueDecl, SourceRange) {
  RangeVerifier<UnresolvedUsingValueDecl> Verifier;
  Verifier.expectRange(3, 3, 3, 6);
  EXPECT_TRUE(Verifier.match(
      "template <typename B>\n"
      "class D : public B {\n"
      "  B::i;\n"
      "};",
      unresolvedUsingValueDecl()));
}

TEST(FriendDecl, FriendNonMemberFunctionLocation) {
  LocationVerifier<FriendDecl> Verifier;
  Verifier.expectLocation(2, 13);
  EXPECT_TRUE(Verifier.match("struct A {\n"
                             "friend void f();\n"
                             "};\n",
                             friendDecl()));
}

TEST(FriendDecl, FriendNonMemberFunctionRange) {
  RangeVerifier<FriendDecl> Verifier;
  Verifier.expectRange(2, 1, 2, 15);
  EXPECT_TRUE(Verifier.match("struct A {\n"
                             "friend void f();\n"
                             "};\n",
                             friendDecl()));
}

TEST(FriendDecl, FriendNonMemberFunctionDefinitionLocation) {
  LocationVerifier<FriendDecl> Verifier;
  Verifier.expectLocation(2, 12);
  EXPECT_TRUE(Verifier.match("struct A {\n"
                             "friend int f() { return 0; }\n"
                             "};\n",
                             friendDecl()));
}

TEST(FriendDecl, FriendNonMemberFunctionDefinitionRange) {
  RangeVerifier<FriendDecl> Verifier;
  Verifier.expectRange(2, 1, 2, 28);
  EXPECT_TRUE(Verifier.match("struct A {\n"
                             "friend int f() { return 0; }\n"
                             "};\n",
                             friendDecl()));
}

TEST(FriendDecl, FriendElaboratedTypeLocation) {
  LocationVerifier<FriendDecl> Verifier;
  Verifier.expectLocation(2, 8);
  EXPECT_TRUE(Verifier.match("struct A {\n"
                             "friend class B;\n"
                             "};\n",
                             friendDecl()));
}

TEST(FriendDecl, FriendElaboratedTypeRange) {
  RangeVerifier<FriendDecl> Verifier;
  Verifier.expectRange(2, 1, 2, 14);
  EXPECT_TRUE(Verifier.match("struct A {\n"
                             "friend class B;\n"
                             "};\n",
                             friendDecl()));
}

TEST(FriendDecl, FriendSimpleTypeLocation) {
  LocationVerifier<FriendDecl> Verifier;
  Verifier.expectLocation(3, 8);
  EXPECT_TRUE(Verifier.match("class B;\n"
                             "struct A {\n"
                             "friend B;\n"
                             "};\n",
                             friendDecl(), Lang_CXX11));
}

TEST(FriendDecl, FriendSimpleTypeRange) {
  RangeVerifier<FriendDecl> Verifier;
  Verifier.expectRange(3, 1, 3, 8);
  EXPECT_TRUE(Verifier.match("class B;\n"
                             "struct A {\n"
                             "friend B;\n"
                             "};\n",
                             friendDecl(), Lang_CXX11));
}

TEST(FriendDecl, FriendTemplateParameterLocation) {
  LocationVerifier<FriendDecl> Verifier;
  Verifier.expectLocation(3, 8);
  EXPECT_TRUE(Verifier.match("template <typename T>\n"
                             "struct A {\n"
                             "friend T;\n"
                             "};\n",
                             friendDecl(), Lang_CXX11));
}

TEST(FriendDecl, FriendTemplateParameterRange) {
  RangeVerifier<FriendDecl> Verifier;
  Verifier.expectRange(3, 1, 3, 8);
  EXPECT_TRUE(Verifier.match("template <typename T>\n"
                             "struct A {\n"
                             "friend T;\n"
                             "};\n",
                             friendDecl(), Lang_CXX11));
}

TEST(FriendDecl, FriendDecltypeLocation) {
  LocationVerifier<FriendDecl> Verifier;
  Verifier.expectLocation(4, 8);
  EXPECT_TRUE(Verifier.match("struct A;\n"
                             "A foo();\n"
                             "struct A {\n"
                             "friend decltype(foo());\n"
                             "};\n",
                             friendDecl(), Lang_CXX11));
}

TEST(FriendDecl, FriendDecltypeRange) {
  RangeVerifier<FriendDecl> Verifier;
  Verifier.expectRange(4, 1, 4, 8);
  EXPECT_TRUE(Verifier.match("struct A;\n"
                             "A foo();\n"
                             "struct A {\n"
                             "friend decltype(foo());\n"
                             "};\n",
                             friendDecl(), Lang_CXX11));
}

TEST(FriendDecl, FriendConstructorDestructorLocation) {
  const std::string Code = "struct B {\n"
                           "B();\n"
                           "~B();\n"
                           "};\n"
                           "struct A {\n"
                           "friend B::B(), B::~B();\n"
                           "};\n";
  LocationVerifier<FriendDecl> ConstructorVerifier;
  ConstructorVerifier.expectLocation(6, 11);
  EXPECT_TRUE(ConstructorVerifier.match(
      Code, friendDecl(has(cxxConstructorDecl(ofClass(hasName("B")))))));
  LocationVerifier<FriendDecl> DestructorVerifier;
  DestructorVerifier.expectLocation(6, 19);
  EXPECT_TRUE(DestructorVerifier.match(
      Code, friendDecl(has(cxxDestructorDecl(ofClass(hasName("B")))))));
}

TEST(FriendDecl, FriendConstructorDestructorRange) {
  const std::string Code = "struct B {\n"
                           "B();\n"
                           "~B();\n"
                           "};\n"
                           "struct A {\n"
                           "friend B::B(), B::~B();\n"
                           "};\n";
  RangeVerifier<FriendDecl> ConstructorVerifier;
  ConstructorVerifier.expectRange(6, 1, 6, 13);
  EXPECT_TRUE(ConstructorVerifier.match(
      Code, friendDecl(has(cxxConstructorDecl(ofClass(hasName("B")))))));
  RangeVerifier<FriendDecl> DestructorVerifier;
  DestructorVerifier.expectRange(6, 1, 6, 22);
  EXPECT_TRUE(DestructorVerifier.match(
      Code, friendDecl(has(cxxDestructorDecl(ofClass(hasName("B")))))));
}

TEST(FriendDecl, FriendTemplateFunctionLocation) {
  LocationVerifier<FriendDecl> Verifier;
  Verifier.expectLocation(3, 13);
  EXPECT_TRUE(Verifier.match("struct A {\n"
                             "template <typename T>\n"
                             "friend void f();\n"
                             "};\n",
                             friendDecl()));
}

TEST(FriendDecl, FriendTemplateFunctionRange) {
  RangeVerifier<FriendDecl> Verifier;
  Verifier.expectRange(2, 1, 3, 15);
  EXPECT_TRUE(Verifier.match("struct A {\n"
                             "template <typename T>\n"
                             "friend void f();\n"
                             "};\n",
                             friendDecl()));
}

TEST(FriendDecl, FriendTemplateClassLocation) {
  LocationVerifier<FriendDecl> Verifier;
  Verifier.expectLocation(3, 14);
  EXPECT_TRUE(Verifier.match("struct A {\n"
                             "template <typename T>\n"
                             "friend class B;\n"
                             "};\n",
                             friendDecl()));
}

TEST(FriendDecl, FriendTemplateClassRange) {
  RangeVerifier<FriendDecl> Verifier;
  Verifier.expectRange(2, 1, 3, 14);
  EXPECT_TRUE(Verifier.match("struct A {\n"
                             "template <typename T>\n"
                             "friend class B;\n"
                             "};\n",
                             friendDecl()));
}

TEST(FriendDecl, FriendInlineFunctionLocation) {
  LocationVerifier<FriendDecl> Verifier;
  Verifier.expectLocation(2, 19);
  EXPECT_TRUE(Verifier.match("struct A {\n"
                             "int inline friend f() { return 0; }"
                             "};\n",
                             friendDecl()));
}

TEST(FriendDecl, FriendInlineFunctionRange) {
  RangeVerifier<FriendDecl> Verifier;
  Verifier.expectRange(2, 1, 2, 35);
  EXPECT_TRUE(Verifier.match("struct A {\n"
                             "int inline friend f() { return 0; }"
                             "};\n",
                             friendDecl(), Lang_CXX11));
}

TEST(FriendDecl, InstantiationSourceRange) {
  RangeVerifier<FriendDecl> Verifier;
  Verifier.expectRange(4, 3, 4, 35);
  EXPECT_TRUE(Verifier.match(
      "template <typename T> class S;\n"
      "template<class T> void operator+(S<T> x);\n"
      "template<class T> struct S {\n"
      "  friend void operator+<>(S<T> src);\n"
      "};\n"
      "void test(S<double> s) { +s; }",
      friendDecl(hasParent(cxxRecordDecl(isTemplateInstantiation())))));
}

TEST(ObjCMessageExpr, ParenExprRange) {
  RangeVerifier<ParenExpr> Verifier;
  Verifier.expectRange(5, 25, 5, 27);
  EXPECT_TRUE(Verifier.match("struct A { int a; };\n"
                             "@interface B {}\n"
                             "+ (void) f1: (A)arg;\n"
                             "@end\n"
                             "void f2() { A a; [B f1: (a)]; }\n",
                             traverse(TK_AsIs, parenExpr()), Lang_OBJCXX));
}

TEST(FunctionDecl, FunctionDeclWithThrowSpecification) {
  RangeVerifier<FunctionDecl> Verifier;
  Verifier.expectRange(1, 1, 1, 16);
  EXPECT_TRUE(Verifier.match(
      "void f() throw();\n",
      functionDecl()));
}

TEST(FunctionDecl, FunctionDeclWithNoExceptSpecification) {
  RangeVerifier<FunctionDecl> Verifier;
  Verifier.expectRange(1, 1, 1, 24);
  EXPECT_TRUE(Verifier.match("void f() noexcept(false);\n", functionDecl(),
                             Lang_CXX11));
}

class FunctionDeclParametersRangeVerifier : public RangeVerifier<FunctionDecl> {
protected:
  SourceRange getRange(const FunctionDecl &Function) override {
    return Function.getParametersSourceRange();
  }
};

TEST(FunctionDeclParameters, FunctionDeclOnlyVariadic) {
  FunctionDeclParametersRangeVerifier Verifier;
  Verifier.expectRange(1, 8, 1, 8);
  EXPECT_TRUE(Verifier.match("void f(...);\n", functionDecl()));
}

TEST(FunctionDeclParameters, FunctionDeclVariadic) {
  FunctionDeclParametersRangeVerifier Verifier;
  Verifier.expectRange(1, 8, 1, 15);
  EXPECT_TRUE(Verifier.match("void f(int a, ...);\n", functionDecl()));
}

TEST(FunctionDeclParameters, FunctionDeclMacroVariadic) {
  FunctionDeclParametersRangeVerifier Verifier;
  Verifier.expectRange(2, 8, 1, 18);
  EXPECT_TRUE(Verifier.match("#define VARIADIC ...\n"
                             "void f(int a, VARIADIC);\n",
                             functionDecl()));
}

TEST(FunctionDeclParameters, FunctionDeclMacroParams) {
  FunctionDeclParametersRangeVerifier Verifier;
  Verifier.expectRange(1, 16, 2, 20);
  EXPECT_TRUE(Verifier.match("#define PARAMS int a, int b\n"
                             "void f(PARAMS, int c);",
                             functionDecl()));
}

TEST(FunctionDeclParameters, FunctionDeclSingleParameter) {
  FunctionDeclParametersRangeVerifier Verifier;
  Verifier.expectRange(1, 8, 1, 12);
  EXPECT_TRUE(Verifier.match("void f(int a);\n", functionDecl()));
}

TEST(FunctionDeclParameters, MemberFunctionDecl) {
  FunctionDeclParametersRangeVerifier Verifier;
  Verifier.expectRange(2, 8, 2, 12);
  EXPECT_TRUE(Verifier.match("class A{\n"
                             "void f(int a);\n"
                             "};",
                             functionDecl()));
}

TEST(FunctionDeclParameters, MemberFunctionDeclVariadic) {
  FunctionDeclParametersRangeVerifier Verifier;
  Verifier.expectRange(2, 8, 2, 15);
  EXPECT_TRUE(Verifier.match("class A{\n"
                             "void f(int a, ...);\n"
                             "};",
                             functionDecl()));
}

TEST(FunctionDeclParameters, StaticFunctionDecl) {
  FunctionDeclParametersRangeVerifier Verifier;
  Verifier.expectRange(2, 15, 2, 19);
  EXPECT_TRUE(Verifier.match("class A{\n"
                             "static void f(int a);\n"
                             "};",
                             functionDecl()));
}

TEST(FunctionDeclParameters, FunctionDeclMultipleParameters) {
  FunctionDeclParametersRangeVerifier Verifier;
  Verifier.expectRange(1, 8, 1, 28);
  EXPECT_TRUE(
      Verifier.match("void f(int a, int b, char *c);\n", functionDecl()));
}

TEST(FunctionDeclParameters, FunctionDeclWithDefaultValue) {
  FunctionDeclParametersRangeVerifier Verifier;
  Verifier.expectRange(1, 8, 1, 16);
  EXPECT_TRUE(Verifier.match("void f(int a = 5);\n", functionDecl()));
}

TEST(FunctionDeclParameters, FunctionDeclWithVolatile) {
  FunctionDeclParametersRangeVerifier Verifier;
  Verifier.expectRange(1, 8, 1, 22);
  EXPECT_TRUE(Verifier.match("void f(volatile int *i);", functionDecl()));
}

TEST(FunctionDeclParameters, FunctionDeclWithConstParam) {
  FunctionDeclParametersRangeVerifier Verifier;
  Verifier.expectRange(1, 8, 1, 19);
  EXPECT_TRUE(Verifier.match("void f(const int *i);", functionDecl()));
}

TEST(FunctionDeclParameters, FunctionDeclWithConstVolatileParam) {
  FunctionDeclParametersRangeVerifier Verifier;
  Verifier.expectRange(1, 8, 1, 28);
  EXPECT_TRUE(Verifier.match("void f(const volatile int *i);", functionDecl()));
}

TEST(FunctionDeclParameters, FunctionDeclWithParamAttribute) {
  FunctionDeclParametersRangeVerifier Verifier;
  Verifier.expectRange(1, 8, 1, 36);
  EXPECT_TRUE(Verifier.match("void f(__attribute__((unused)) int a) {}",
                             functionDecl()));
}

TEST(CXXMethodDecl, CXXMethodDeclWithThrowSpecification) {
  RangeVerifier<FunctionDecl> Verifier;
  Verifier.expectRange(2, 1, 2, 16);
  EXPECT_TRUE(Verifier.match(
      "class A {\n"
      "void f() throw();\n"
      "};\n",
      functionDecl()));
}

TEST(CXXMethodDecl, CXXMethodDeclWithNoExceptSpecification) {
  RangeVerifier<FunctionDecl> Verifier;
  Verifier.expectRange(2, 1, 2, 24);
  EXPECT_TRUE(Verifier.match("class A {\n"
                             "void f() noexcept(false);\n"
                             "};\n",
                             functionDecl(), Lang_CXX11));
}

class ExceptionSpecRangeVerifier : public RangeVerifier<TypeLoc> {
protected:
  SourceRange getRange(const TypeLoc &Node) override {
    auto T =
      Node.getUnqualifiedLoc().castAs<FunctionProtoTypeLoc>();
    assert(!T.isNull());
    return T.getExceptionSpecRange();
  }
};

class ParmVarExceptionSpecRangeVerifier : public RangeVerifier<ParmVarDecl> {
protected:
  SourceRange getRange(const ParmVarDecl &Node) override {
    if (const TypeSourceInfo *TSI = Node.getTypeSourceInfo()) {
      TypeLoc TL = TSI->getTypeLoc();
      if (TL.getType()->isPointerType()) {
        TL = TL.getNextTypeLoc().IgnoreParens();
        if (auto FPTL = TL.getAs<FunctionProtoTypeLoc>()) {
          return FPTL.getExceptionSpecRange();
        }
      }
    }
    return SourceRange();
  }
};

TEST(FunctionDecl, ExceptionSpecifications) {
  ExceptionSpecRangeVerifier Verifier;

  Verifier.expectRange(1, 10, 1, 16);
  EXPECT_TRUE(Verifier.match("void f() throw();\n", loc(functionType())));

  Verifier.expectRange(1, 10, 1, 34);
  EXPECT_TRUE(Verifier.match("void f() throw(void(void) throw());\n",
                             loc(functionType())));

  Verifier.expectRange(1, 10, 1, 19);
  std::vector<std::string> Args;
  Args.push_back("-fms-extensions");
  EXPECT_TRUE(Verifier.match("void f() throw(...);\n", loc(functionType()),
                             Args, Lang_CXX03));

  Verifier.expectRange(1, 10, 1, 10);
  EXPECT_TRUE(
      Verifier.match("void f() noexcept;\n", loc(functionType()), Lang_CXX11));

  Verifier.expectRange(1, 10, 1, 24);
  EXPECT_TRUE(Verifier.match("void f() noexcept(false);\n", loc(functionType()),
                             Lang_CXX11));

  Verifier.expectRange(1, 10, 1, 32);
  EXPECT_TRUE(Verifier.match("void f() noexcept(noexcept(1+1));\n",
                             loc(functionType()), Lang_CXX11));

  ParmVarExceptionSpecRangeVerifier Verifier2;
  Verifier2.expectRange(1, 25, 1, 31);
  EXPECT_TRUE(Verifier2.match("void g(void (*fp)(void) throw());\n",
                              parmVarDecl(hasType(pointerType(pointee(
                                  parenType(innerType(functionType()))))))));

  Verifier2.expectRange(1, 25, 1, 38);
  EXPECT_TRUE(Verifier2.match("void g(void (*fp)(void) noexcept(true));\n",
                              parmVarDecl(hasType(pointerType(pointee(
                                  parenType(innerType(functionType())))))),
                              Lang_CXX11));
}

TEST(Decl, MemberPointerStarLoc) {
  llvm::Annotations Example(R"cpp(
    struct X {};
    int X::$star^* a;
  )cpp");

  auto AST = tooling::buildASTFromCode(Example.code());
  SourceManager &SM = AST->getSourceManager();
  auto &Ctx = AST->getASTContext();

  auto *VD = selectFirst<VarDecl>("vd", match(varDecl().bind("vd"), Ctx));
  ASSERT_TRUE(VD != nullptr);

  auto TL =
      VD->getTypeSourceInfo()->getTypeLoc().castAs<MemberPointerTypeLoc>();
  ASSERT_EQ(SM.getFileOffset(TL.getStarLoc()), Example.point("star"));
}

} // end namespace
