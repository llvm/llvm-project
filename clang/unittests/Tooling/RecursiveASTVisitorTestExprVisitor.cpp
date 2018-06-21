//===- unittest/Tooling/RecursiveASTVisitorTestExprVisitor.cpp ------------===//
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

class ParenExprVisitor : public ExpectedLocationVisitor<ParenExprVisitor> {
public:
  bool VisitParenExpr(ParenExpr *Parens) {
    Match("", Parens->getExprLoc());
    return true;
  }
};

TEST(RecursiveASTVisitor, VisitsParensDuringDataRecursion) {
  ParenExprVisitor Visitor;
  Visitor.ExpectMatch("", 1, 9);
  EXPECT_TRUE(Visitor.runOver("int k = (4) + 9;\n"));
}

class TemplateArgumentLocTraverser
  : public ExpectedLocationVisitor<TemplateArgumentLocTraverser> {
public:
  bool TraverseTemplateArgumentLoc(const TemplateArgumentLoc &ArgLoc) {
    std::string ArgStr;
    llvm::raw_string_ostream Stream(ArgStr);
    const TemplateArgument &Arg = ArgLoc.getArgument();

    Arg.print(Context->getPrintingPolicy(), Stream);
    Match(Stream.str(), ArgLoc.getLocation());
    return ExpectedLocationVisitor<TemplateArgumentLocTraverser>::
      TraverseTemplateArgumentLoc(ArgLoc);
  }
};

TEST(RecursiveASTVisitor, VisitsClassTemplateTemplateParmDefaultArgument) {
  TemplateArgumentLocTraverser Visitor;
  Visitor.ExpectMatch("X", 2, 40);
  EXPECT_TRUE(Visitor.runOver(
    "template<typename T> class X;\n"
    "template<template <typename> class T = X> class Y;\n"
    "template<template <typename> class T> class Y {};\n"));
}

class CXXBoolLiteralExprVisitor 
  : public ExpectedLocationVisitor<CXXBoolLiteralExprVisitor> {
public:
  bool VisitCXXBoolLiteralExpr(CXXBoolLiteralExpr *BE) {
    if (BE->getValue())
      Match("true", BE->getLocation());
    else
      Match("false", BE->getLocation());
    return true;
  }
};

TEST(RecursiveASTVisitor, VisitsClassTemplateNonTypeParmDefaultArgument) {
  CXXBoolLiteralExprVisitor Visitor;
  Visitor.ExpectMatch("true", 2, 19);
  EXPECT_TRUE(Visitor.runOver(
    "template<bool B> class X;\n"
    "template<bool B = true> class Y;\n"
    "template<bool B> class Y {};\n"));
}

// A visitor that visits implicit declarations and matches constructors.
class ImplicitCtorVisitor
    : public ExpectedLocationVisitor<ImplicitCtorVisitor> {
public:
  bool shouldVisitImplicitCode() const { return true; }

  bool VisitCXXConstructorDecl(CXXConstructorDecl* Ctor) {
    if (Ctor->isImplicit()) {  // Was not written in source code
      if (const CXXRecordDecl* Class = Ctor->getParent()) {
        Match(Class->getName(), Ctor->getLocation());
      }
    }
    return true;
  }
};

TEST(RecursiveASTVisitor, VisitsImplicitCopyConstructors) {
  ImplicitCtorVisitor Visitor;
  Visitor.ExpectMatch("Simple", 2, 8);
  // Note: Clang lazily instantiates implicit declarations, so we need
  // to use them in order to force them to appear in the AST.
  EXPECT_TRUE(Visitor.runOver(
      "struct WithCtor { WithCtor(); }; \n"
      "struct Simple { Simple(); WithCtor w; }; \n"
      "int main() { Simple s; Simple t(s); }\n"));
}

/// \brief A visitor that optionally includes implicit code and matches
/// CXXConstructExpr.
///
/// The name recorded for the match is the name of the class whose constructor
/// is invoked by the CXXConstructExpr, not the name of the class whose
/// constructor the CXXConstructExpr is contained in.
class ConstructExprVisitor
    : public ExpectedLocationVisitor<ConstructExprVisitor> {
public:
  ConstructExprVisitor() : ShouldVisitImplicitCode(false) {}

  bool shouldVisitImplicitCode() const { return ShouldVisitImplicitCode; }

  void setShouldVisitImplicitCode(bool NewValue) {
    ShouldVisitImplicitCode = NewValue;
  }

  bool VisitCXXConstructExpr(CXXConstructExpr* Expr) {
    if (const CXXConstructorDecl* Ctor = Expr->getConstructor()) {
      if (const CXXRecordDecl* Class = Ctor->getParent()) {
        Match(Class->getName(), Expr->getLocation());
      }
    }
    return true;
  }

 private:
  bool ShouldVisitImplicitCode;
};

TEST(RecursiveASTVisitor, CanVisitImplicitMemberInitializations) {
  ConstructExprVisitor Visitor;
  Visitor.setShouldVisitImplicitCode(true);
  Visitor.ExpectMatch("WithCtor", 2, 8);
  // Simple has a constructor that implicitly initializes 'w'.  Test
  // that a visitor that visits implicit code visits that initialization.
  // Note: Clang lazily instantiates implicit declarations, so we need
  // to use them in order to force them to appear in the AST.
  EXPECT_TRUE(Visitor.runOver(
      "struct WithCtor { WithCtor(); }; \n"
      "struct Simple { WithCtor w; }; \n"
      "int main() { Simple s; }\n"));
}

// The same as CanVisitImplicitMemberInitializations, but checking that the
// visits are omitted when the visitor does not include implicit code.
TEST(RecursiveASTVisitor, CanSkipImplicitMemberInitializations) {
  ConstructExprVisitor Visitor;
  Visitor.setShouldVisitImplicitCode(false);
  Visitor.DisallowMatch("WithCtor", 2, 8);
  // Simple has a constructor that implicitly initializes 'w'.  Test
  // that a visitor that skips implicit code skips that initialization.
  // Note: Clang lazily instantiates implicit declarations, so we need
  // to use them in order to force them to appear in the AST.
  EXPECT_TRUE(Visitor.runOver(
      "struct WithCtor { WithCtor(); }; \n"
      "struct Simple { WithCtor w; }; \n"
      "int main() { Simple s; }\n"));
}

class DeclRefExprVisitor : public ExpectedLocationVisitor<DeclRefExprVisitor> {
public:
  DeclRefExprVisitor() : ShouldVisitImplicitCode(false) {}

  bool shouldVisitImplicitCode() const { return ShouldVisitImplicitCode; }

  void setShouldVisitImplicitCode(bool NewValue) {
    ShouldVisitImplicitCode = NewValue;
  }

  bool VisitDeclRefExpr(DeclRefExpr *Reference) {
    Match(Reference->getNameInfo().getAsString(), Reference->getLocation());
    return true;
  }

private:
  bool ShouldVisitImplicitCode;
};

TEST(RecursiveASTVisitor, VisitsBaseClassTemplateArguments) {
  DeclRefExprVisitor Visitor;
  Visitor.ExpectMatch("x", 2, 3);
  EXPECT_TRUE(Visitor.runOver(
    "void x(); template <void (*T)()> class X {};\nX<x> y;"));
}

TEST(RecursiveASTVisitor, VisitsCXXForRangeStmtRange) {
  DeclRefExprVisitor Visitor;
  Visitor.ExpectMatch("x", 2, 25);
  Visitor.ExpectMatch("x", 2, 30);
  EXPECT_TRUE(Visitor.runOver(
    "int x[5];\n"
    "void f() { for (int i : x) { x[0] = 1; } }",
    DeclRefExprVisitor::Lang_CXX11));
}

TEST(RecursiveASTVisitor, VisitsCallExpr) {
  DeclRefExprVisitor Visitor;
  Visitor.ExpectMatch("x", 1, 22);
  EXPECT_TRUE(Visitor.runOver(
    "void x(); void y() { x(); }"));
}

TEST(RecursiveASTVisitor, VisitsExplicitLambdaCaptureInit) {
  DeclRefExprVisitor Visitor;
  Visitor.ExpectMatch("i", 1, 20);
  EXPECT_TRUE(Visitor.runOver(
    "void f() { int i; [i]{}; }",
    DeclRefExprVisitor::Lang_CXX11));
}

TEST(RecursiveASTVisitor, VisitsUseOfImplicitLambdaCapture) {
  DeclRefExprVisitor Visitor;
  Visitor.ExpectMatch("i", 1, 24);
  EXPECT_TRUE(Visitor.runOver(
    "void f() { int i; [=]{ i; }; }",
    DeclRefExprVisitor::Lang_CXX11));
}

TEST(RecursiveASTVisitor, VisitsImplicitLambdaCaptureInit) {
  DeclRefExprVisitor Visitor;
  Visitor.setShouldVisitImplicitCode(true);
  // We're expecting the "i" in the lambda to be visited twice:
  // - Once for the DeclRefExpr in the lambda capture initialization (whose
  //   source code location is set to the first use of the variable).
  // - Once for the DeclRefExpr for the use of "i" inside the lambda.
  Visitor.ExpectMatch("i", 1, 24, /*Times=*/2);
  EXPECT_TRUE(Visitor.runOver(
    "void f() { int i; [=]{ i; }; }",
    DeclRefExprVisitor::Lang_CXX11));
}

TEST(RecursiveASTVisitor, VisitsLambdaInitCaptureInit) {
  DeclRefExprVisitor Visitor;
  Visitor.ExpectMatch("i", 1, 24);
  EXPECT_TRUE(Visitor.runOver(
    "void f() { int i; [a = i + 1]{}; }",
    DeclRefExprVisitor::Lang_CXX14));
}

/* FIXME: According to Richard Smith this is a bug in the AST.
TEST(RecursiveASTVisitor, VisitsBaseClassTemplateArgumentsInInstantiation) {
  DeclRefExprVisitor Visitor;
  Visitor.ExpectMatch("x", 3, 43);
  EXPECT_TRUE(Visitor.runOver(
    "template <typename T> void x();\n"
    "template <void (*T)()> class X {};\n"
    "template <typename T> class Y : public X< x<T> > {};\n"
    "Y<int> y;"));
}
*/

TEST(RecursiveASTVisitor, VisitsExtension) {
  DeclRefExprVisitor Visitor;
  Visitor.ExpectMatch("s", 1, 24);
  EXPECT_TRUE(Visitor.runOver(
    "int s = __extension__ (s);\n"));
}

TEST(RecursiveASTVisitor, VisitsCopyExprOfBlockDeclCapture) {
  DeclRefExprVisitor Visitor;
  Visitor.ExpectMatch("x", 3, 24);
  EXPECT_TRUE(Visitor.runOver("void f(int(^)(int)); \n"
                              "void g() { \n"
                              "  f([&](int x){ return x; }); \n"
                              "}",
                              DeclRefExprVisitor::Lang_OBJCXX11));
}

} // end anonymous namespace
