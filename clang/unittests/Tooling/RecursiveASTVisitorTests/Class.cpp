//===- unittest/Tooling/RecursiveASTVisitorTests/Class.cpp ----------------===//
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

} // end anonymous namespace
