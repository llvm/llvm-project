//===- unittest/Tooling/RecursiveASTVisitorTestTypeLocVisitor.cpp ---------===//
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

class TypeLocVisitor : public ExpectedLocationVisitor<TypeLocVisitor> {
public:
  bool VisitTypeLoc(TypeLoc TypeLocation) {
    Match(TypeLocation.getType().getAsString(), TypeLocation.getBeginLoc());
    return true;
  }
};

TEST(RecursiveASTVisitor, VisitsBaseClassDeclarations) {
  TypeLocVisitor Visitor;
  Visitor.ExpectMatch("class X", 1, 30);
  EXPECT_TRUE(Visitor.runOver("class X {}; class Y : public X {};"));
}

TEST(RecursiveASTVisitor, VisitsCXXBaseSpecifiersOfForwardDeclaredClass) {
  TypeLocVisitor Visitor;
  Visitor.ExpectMatch("class X", 3, 18);
  EXPECT_TRUE(Visitor.runOver(
    "class Y;\n"
    "class X {};\n"
    "class Y : public X {};"));
}

TEST(RecursiveASTVisitor, VisitsCXXBaseSpecifiersWithIncompleteInnerClass) {
  TypeLocVisitor Visitor;
  Visitor.ExpectMatch("class X", 2, 18);
  EXPECT_TRUE(Visitor.runOver(
    "class X {};\n"
    "class Y : public X { class Z; };"));
}

TEST(RecursiveASTVisitor, VisitsCXXBaseSpecifiersOfSelfReferentialType) {
  TypeLocVisitor Visitor;
  Visitor.ExpectMatch("X<class Y>", 2, 18);
  EXPECT_TRUE(Visitor.runOver(
    "template<typename T> class X {};\n"
    "class Y : public X<Y> {};"));
}

TEST(RecursiveASTVisitor, VisitsClassTemplateTypeParmDefaultArgument) {
  TypeLocVisitor Visitor;
  Visitor.ExpectMatch("class X", 2, 23);
  EXPECT_TRUE(Visitor.runOver(
    "class X;\n"
    "template<typename T = X> class Y;\n"
    "template<typename T> class Y {};\n"));
}

TEST(RecursiveASTVisitor, VisitsCompoundLiteralType) {
  TypeLocVisitor Visitor;
  Visitor.ExpectMatch("struct S", 1, 26);
  EXPECT_TRUE(Visitor.runOver(
      "int f() { return (struct S { int a; }){.a = 0}.a; }",
      TypeLocVisitor::Lang_C));
}

TEST(RecursiveASTVisitor, VisitsObjCPropertyType) {
  TypeLocVisitor Visitor;
  Visitor.ExpectMatch("NSNumber", 2, 33);
  EXPECT_TRUE(Visitor.runOver(
      "@class NSNumber; \n"
      "@interface A @property (retain) NSNumber *x; @end\n",
      TypeLocVisitor::Lang_OBJC));
}

TEST(RecursiveASTVisitor, VisitInvalidType) {
  TypeLocVisitor Visitor;
  // FIXME: It would be nice to have information about subtypes of invalid type
  //Visitor.ExpectMatch("typeof(struct F *) []", 1, 1);
  // Even if the full type is invalid, it should still find sub types
  //Visitor.ExpectMatch("struct F", 1, 19);
  EXPECT_FALSE(Visitor.runOver(
      "__typeof__(struct F*) var[invalid];\n",
      TypeLocVisitor::Lang_C));
}

} // end anonymous namespace
