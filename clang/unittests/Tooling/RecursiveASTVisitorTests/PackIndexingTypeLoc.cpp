//===- unittest/Tooling/RecursiveASTVisitorTests/PackIndexingTypeLoc.cpp --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestVisitor.h"

using namespace clang;

namespace {

class PackIndexingTypeLocVisitor : public ExpectedLocationVisitor {
public:
  bool VisitTemplateTypeParmTypeLoc(TemplateTypeParmTypeLoc TL) override {
    if (!TL)
      return true;
    Match(TL.getDecl()->getName(), TL.getNameLoc());
    return true;
  }
};

TEST(RecursiveASTVisitor, VisitTypeLocInPackIndexingType) {
  PackIndexingTypeLocVisitor Visitor;
  Visitor.ExpectMatch("T", 4, 8);
  llvm::StringLiteral Code = R"cpp(
     template <typename... T>
     void foo() {
       T...[0] x;
     }
  )cpp";
  EXPECT_TRUE(Visitor.runOver(Code, PackIndexingTypeLocVisitor::Lang_CXX26));
}

TEST(RecursiveASTVisitor, VisitTypeLocInPackIndexingTypeAlias) {
  PackIndexingTypeLocVisitor Visitor;
  Visitor.ExpectMatch("T", 4, 18);
  llvm::StringLiteral Code = R"cpp(
     template <typename... T>
     void foo() {
       using U = T...[0];
     }
  )cpp";
  EXPECT_TRUE(Visitor.runOver(Code, PackIndexingTypeLocVisitor::Lang_CXX26));
}
} // end anonymous namespace
