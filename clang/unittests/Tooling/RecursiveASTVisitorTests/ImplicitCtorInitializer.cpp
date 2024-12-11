//=- unittest/Tooling/RecursiveASTVisitorTests/ImplicitCtorInitializer.cpp -=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestVisitor.h"

using namespace clang;

namespace {

class CXXCtorInitializerVisitor
    : public ExpectedLocationVisitor<CXXCtorInitializerVisitor> {
public:
  CXXCtorInitializerVisitor(bool VisitImplicitCode)
      : VisitImplicitCode(VisitImplicitCode) {}

  bool shouldVisitImplicitCode() const { return VisitImplicitCode; }

  bool TraverseConstructorInitializer(CXXCtorInitializer *Init) {
    if (!Init->isWritten())
      VisitedImplicitInitializer = true;
    Match("initializer", Init->getSourceLocation());
    return ExpectedLocationVisitor<
        CXXCtorInitializerVisitor>::TraverseConstructorInitializer(Init);
  }

  bool VisitedImplicitInitializer = false;

private:
  bool VisitImplicitCode;
};

// Check to ensure that CXXCtorInitializer is not visited when implicit code
// should not be visited and that it is visited when implicit code should be
// visited.
TEST(RecursiveASTVisitor, CXXCtorInitializerVisitNoImplicit) {
  for (bool VisitImplCode : {true, false}) {
    CXXCtorInitializerVisitor Visitor(VisitImplCode);
    Visitor.ExpectMatch("initializer", 7, 17);
    llvm::StringRef Code = R"cpp(
        class A {};
        class B : public A {
          B() {};
        };
        class C : public A {
          C() : A() {}
        };
      )cpp";
    EXPECT_TRUE(Visitor.runOver(Code, CXXCtorInitializerVisitor::Lang_CXX));
    EXPECT_EQ(Visitor.VisitedImplicitInitializer, VisitImplCode);
  }
}
} // end anonymous namespace
