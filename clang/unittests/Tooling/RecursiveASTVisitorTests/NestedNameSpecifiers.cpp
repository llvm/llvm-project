//===- unittest/Tooling/RecursiveASTVisitorTests/NestedNameSpecifiers.cpp -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestVisitor.h"

using namespace clang;

namespace {

// Check to ensure that nested name specifiers are visited.
class NestedNameSpecifiersVisitor : public ExpectedLocationVisitor {
public:
  bool VisitRecordTypeLoc(RecordTypeLoc RTL) override {
    if (!RTL)
      return true;
    Match(RTL.getDecl()->getName(), RTL.getNameLoc());
    return true;
  }

  bool TraverseNestedNameSpecifierLoc(NestedNameSpecifierLoc NNS) override {
    if (!NNS)
      return true;
    if (const NamespaceDecl *ND =
            NNS.getNestedNameSpecifier()->getAsNamespace())
      Match(ND->getName(), NNS.getLocalBeginLoc());
    return ExpectedLocationVisitor::TraverseNestedNameSpecifierLoc(NNS);
  }
};

TEST(RecursiveASTVisitor,
     NestedNameSpecifiersForTemplateSpecializationsAreVisited) {
  StringRef Source = R"(
namespace ns {
struct Outer {
    template<typename T, typename U>
    struct Nested { };

    template<typename T>
    static T x;
};
}

template<>
struct ns::Outer::Nested<int, int>;

template<>
struct ns::Outer::Nested<int, int> { };

template<typename T>
struct ns::Outer::Nested<int, T> { };

template<>
int ns::Outer::x<int> = 0;
)";
  NestedNameSpecifiersVisitor Visitor;
  Visitor.ExpectMatch("ns", 13, 8);
  Visitor.ExpectMatch("ns", 16, 8);
  Visitor.ExpectMatch("ns", 19, 8);
  Visitor.ExpectMatch("ns", 22, 5);
  Visitor.ExpectMatch("Outer", 13, 12);
  Visitor.ExpectMatch("Outer", 16, 12);
  Visitor.ExpectMatch("Outer", 19, 12);
  Visitor.ExpectMatch("Outer", 22, 9);
  EXPECT_TRUE(Visitor.runOver(Source, NestedNameSpecifiersVisitor::Lang_CXX14));
}

} // end anonymous namespace
