//===- unittest/Tooling/RecursiveASTVisitorTests/MemberPointerTypeLoc.cpp -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestVisitor.h"
#include "llvm/ADT/StringRef.h"

using namespace clang;

namespace {

class MemberPointerTypeLocVisitor
    : public ExpectedLocationVisitor<MemberPointerTypeLocVisitor> {
public:
  bool VisitTemplateTypeParmTypeLoc(TemplateTypeParmTypeLoc TL) {
    if (!TL)
      return true;
    Match(TL.getDecl()->getName(), TL.getNameLoc());
    return true;
  }
  bool VisitRecordTypeLoc(RecordTypeLoc RTL) {
    if (!RTL)
      return true;
    Match(RTL.getDecl()->getName(), RTL.getNameLoc());
    return true;
  }
};

TEST(RecursiveASTVisitor, VisitTypeLocInMemberPointerTypeLoc) {
  MemberPointerTypeLocVisitor Visitor;
  Visitor.ExpectMatch("Bar", 4, 36);
  Visitor.ExpectMatch("T", 7, 23);
  llvm::StringLiteral Code = R"cpp(
     class Bar { void func(int); };
     class Foo {
       void bind(const char*, void(Bar::*Foo)(int)) {}

       template<typename T>
       void test(void(T::*Foo)());
     };
  )cpp";
  EXPECT_TRUE(Visitor.runOver(Code));
}

TEST(RecursiveASTVisitor, NoCrash) {
  MemberPointerTypeLocVisitor Visitor;
  llvm::StringLiteral Code = R"cpp(
     // MemberPointerTypeLoc.getClassTInfo() is null.
     class a(b(a::*)) class
  )cpp";
  EXPECT_FALSE(Visitor.runOver(Code));
}

} // end anonymous namespace
