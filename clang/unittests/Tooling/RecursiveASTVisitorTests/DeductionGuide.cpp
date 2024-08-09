//===- unittest/Tooling/RecursiveASTVisitorTests/DeductionGuide.cpp -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestVisitor.h"
#include <string>

using namespace clang;

namespace {

class DeductionGuideVisitor
    : public ExpectedLocationVisitor<DeductionGuideVisitor> {
public:
  DeductionGuideVisitor(bool ShouldVisitImplicitCode)
      : ShouldVisitImplicitCode(ShouldVisitImplicitCode) {}
  bool VisitCXXDeductionGuideDecl(CXXDeductionGuideDecl *D) {
    std::string Storage;
    llvm::raw_string_ostream Stream(Storage);
    D->print(Stream);
    Match(Storage, D->getLocation());
    return true;
  }

  bool shouldVisitTemplateInstantiations() const { return false; }

  bool shouldVisitImplicitCode() const { return ShouldVisitImplicitCode; }
  bool ShouldVisitImplicitCode;
};

TEST(RecursiveASTVisitor, DeductionGuideNonImplicitMode) {
  DeductionGuideVisitor Visitor(/*ShouldVisitImplicitCode*/ false);
  // Verify that the synthezied deduction guide for alias is not visited in
  // RAV's implicit mode.
  Visitor.ExpectMatch("Foo(T) -> Foo<int>", 11, 1);
  Visitor.DisallowMatch("Bar(T) -> Foo<int>", 14, 1);
  EXPECT_TRUE(Visitor.runOver(
      R"cpp(
template <typename T>
concept False = true;

template <typename T> 
struct Foo { 
  Foo(T);
};

template<typename T> requires False<T>
Foo(T) -> Foo<int>;

template <typename U>
using Bar = Foo<U>;
Bar s(1); 
   )cpp",
      DeductionGuideVisitor::Lang_CXX2a));
}

TEST(RecursiveASTVisitor, DeductionGuideImplicitMode) {
  DeductionGuideVisitor Visitor(/*ShouldVisitImplicitCode*/ true);
  Visitor.ExpectMatch("Foo(T) -> Foo<int>", 11, 1);
  Visitor.ExpectMatch("Bar(T) -> Foo<int>", 14, 1);
  EXPECT_TRUE(Visitor.runOver(
      R"cpp(
template <typename T>
concept False = true;

template <typename T> 
struct Foo { 
  Foo(T);
};

template<typename T> requires False<T>
Foo(T) -> Foo<int>;

template <typename U>
using Bar = Foo<U>;
Bar s(1); 
   )cpp",
      DeductionGuideVisitor::Lang_CXX2a));
}

} // end anonymous namespace
