//===-- ShowSelectionTreeTests.cpp ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TweakTesting.h"
#include "gmock/gmock-matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {

TWEAK_TEST(ShowSelectionTree);

TEST_F(ShowSelectionTreeTest, Test) {
  EXPECT_AVAILABLE("^int f^oo() { re^turn 2 ^+ 2; }");
  EXPECT_AVAILABLE("/*c^omment*/ int foo() { return 2 ^ + 2; }");

  const char *Output = R"(message:
 TranslationUnitDecl 
   VarDecl int x = fcall(2 + 2)
    .CallExpr fcall(2 + 2)
       ImplicitCastExpr fcall
        .DeclRefExpr fcall
      .BinaryOperator 2 + 2
        *IntegerLiteral 2
)";
  EXPECT_EQ(apply("int fcall(int); int x = fca[[ll(2 +]]2);"), Output);

  Output = R"(message:
 TranslationUnitDecl 
   FunctionDecl void x()
     CompoundStmt { …
       ForStmt for (;;) …
        *BreakStmt break;
)";
  EXPECT_EQ(apply("void x() { for (;;) br^eak; }"), Output);
}

} // namespace
} // namespace clangd
} // namespace clang
