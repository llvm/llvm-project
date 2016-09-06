//===- unittest/Tooling/RecursiveASTVisitorTest.cpp -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "TestVisitor.h"
#include <stack>

using namespace clang;

namespace {

class LambdaExprVisitor : public ExpectedLocationVisitor<LambdaExprVisitor> {
public:
  bool VisitLambdaExpr(LambdaExpr *Lambda) {
    PendingBodies.push(Lambda);
    Match("", Lambda->getIntroducerRange().getBegin());
    return true;
  }
  /// For each call to VisitLambdaExpr, we expect a subsequent call (with
  /// proper nesting) to TraverseLambdaBody.
  bool TraverseLambdaBody(LambdaExpr *Lambda) {
    EXPECT_FALSE(PendingBodies.empty());
    EXPECT_EQ(PendingBodies.top(), Lambda);
    PendingBodies.pop();
    return TraverseStmt(Lambda->getBody());
  }
  /// Determine whether TraverseLambdaBody has been called for every call to
  /// VisitLambdaExpr.
  bool allBodiesHaveBeenTraversed() const {
    return PendingBodies.empty();
  }
private:
  std::stack<LambdaExpr *> PendingBodies;
};

TEST(RecursiveASTVisitor, VisitsLambdaExpr) {
  LambdaExprVisitor Visitor;
  Visitor.ExpectMatch("", 1, 12);
  EXPECT_TRUE(Visitor.runOver("void f() { []{ return; }(); }",
                              LambdaExprVisitor::Lang_CXX11));
}

TEST(RecursiveASTVisitor, TraverseLambdaBodyCanBeOverridden) {
  LambdaExprVisitor Visitor;
  EXPECT_TRUE(Visitor.runOver("void f() { []{ return; }(); }",
                              LambdaExprVisitor::Lang_CXX11));
  EXPECT_TRUE(Visitor.allBodiesHaveBeenTraversed());
}

// Matches the (optional) capture-default of a lambda-introducer.
class LambdaDefaultCaptureVisitor
  : public ExpectedLocationVisitor<LambdaDefaultCaptureVisitor> {
public:
  bool VisitLambdaExpr(LambdaExpr *Lambda) {
    if (Lambda->getCaptureDefault() != LCD_None) {
      Match("", Lambda->getCaptureDefaultLoc());
    }
    return true;
  }
};

TEST(RecursiveASTVisitor, HasCaptureDefaultLoc) {
  LambdaDefaultCaptureVisitor Visitor;
  Visitor.ExpectMatch("", 1, 20);
  EXPECT_TRUE(Visitor.runOver("void f() { int a; [=]{a;}; }",
                              LambdaDefaultCaptureVisitor::Lang_CXX11));
}

// Checks for lambda classes that are not marked as implicitly-generated.
// (There should be none.)
class ClassVisitor : public ExpectedLocationVisitor<ClassVisitor> {
public:
  ClassVisitor() : SawNonImplicitLambdaClass(false) {}
  bool VisitCXXRecordDecl(CXXRecordDecl* record) {
    if (record->isLambda() && !record->isImplicit())
      SawNonImplicitLambdaClass = true;
    return true;
  }

  bool sawOnlyImplicitLambdaClasses() const {
    return !SawNonImplicitLambdaClass;
  }

private:
  bool SawNonImplicitLambdaClass;
};

TEST(RecursiveASTVisitor, LambdaClosureTypesAreImplicit) {
  ClassVisitor Visitor;
  EXPECT_TRUE(Visitor.runOver("auto lambda = []{};", ClassVisitor::Lang_CXX11));
  EXPECT_TRUE(Visitor.sawOnlyImplicitLambdaClasses());
}


// Check to ensure that attributes and expressions within them are being
// visited.
class AttrVisitor : public ExpectedLocationVisitor<AttrVisitor> {
public:
  bool VisitMemberExpr(MemberExpr *ME) {
    Match(ME->getMemberDecl()->getNameAsString(), ME->getLocStart());
    return true;
  }
  bool VisitAttr(Attr *A) {
    Match("Attr", A->getLocation());
    return true;
  }
  bool VisitGuardedByAttr(GuardedByAttr *A) {
    Match("guarded_by", A->getLocation());
    return true;
  }
};


TEST(RecursiveASTVisitor, AttributesAreVisited) {
  AttrVisitor Visitor;
  Visitor.ExpectMatch("Attr", 4, 24);
  Visitor.ExpectMatch("guarded_by", 4, 24);
  Visitor.ExpectMatch("mu1",  4, 35);
  Visitor.ExpectMatch("Attr", 5, 29);
  Visitor.ExpectMatch("mu1",  5, 54);
  Visitor.ExpectMatch("mu2",  5, 59);
  EXPECT_TRUE(Visitor.runOver(
    "class Foo {\n"
    "  int mu1;\n"
    "  int mu2;\n"
    "  int a __attribute__((guarded_by(mu1)));\n"
    "  void bar() __attribute__((exclusive_locks_required(mu1, mu2)));\n"
    "};\n"));
}

// Check to ensure that VarDecls are visited.
class VarDeclVisitor : public ExpectedLocationVisitor<VarDeclVisitor> {
public:
  bool VisitVarDecl(VarDecl *VD) {
    Match(VD->getNameAsString(), VD->getLocStart());
    return true;
  }
};

TEST(RecursiveASTVisitor, ArrayInitializersAreVisited) {
  VarDeclVisitor Visitor;
  Visitor.ExpectMatch("__i0", 1, 8);
  EXPECT_TRUE(
      Visitor.runOver("struct MyClass {\n"
                      "  int c[1];\n"
                      "  static MyClass Create() { return MyClass(); }\n"
                      "};\n"));
}

} // end anonymous namespace
