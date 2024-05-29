//===- unittest/Tooling/RecursiveASTVisitorTests/BitfieldInitializer.cpp -===//
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

// Check to ensure that bitfield initializers are visited.
class BitfieldInitializerVisitor
    : public ExpectedLocationVisitor<BitfieldInitializerVisitor> {
public:
  bool VisitIntegerLiteral(IntegerLiteral *IL) {
    Match(std::to_string(IL->getValue().getSExtValue()), IL->getLocation());
    return true;
  }
};

TEST(RecursiveASTVisitor, BitfieldInitializerIsVisited) {
  BitfieldInitializerVisitor Visitor;
  Visitor.ExpectMatch("123", 2, 15);
  EXPECT_TRUE(Visitor.runOver("struct S {\n"
                              "  int x : 8 = 123;\n"
                              "};\n"));
}

} // end anonymous namespace
