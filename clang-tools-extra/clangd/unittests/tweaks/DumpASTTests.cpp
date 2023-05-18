//===-- DumpASTTests.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TweakTesting.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using ::testing::AllOf;
using ::testing::HasSubstr;
using ::testing::StartsWith;

namespace clang {
namespace clangd {
namespace {

TWEAK_TEST(DumpAST);

TEST_F(DumpASTTest, Test) {
  EXPECT_AVAILABLE("^int f^oo() { re^turn 2 ^+ 2; }");
  EXPECT_UNAVAILABLE("/*c^omment*/ int foo() { return 2 ^ + 2; }");
  EXPECT_THAT(apply("int x = 2 ^+ 2;"),
              AllOf(StartsWith("message:"), HasSubstr("BinaryOperator"),
                    HasSubstr("'+'"), HasSubstr("|-IntegerLiteral"),
                    HasSubstr("<col:9> 'int' 2\n`-IntegerLiteral"),
                    HasSubstr("<col:13> 'int' 2")));
}

} // namespace
} // namespace clangd
} // namespace clang
