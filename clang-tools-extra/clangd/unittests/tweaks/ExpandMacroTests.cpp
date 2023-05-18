//===-- ExpandMacroTests.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TweakTesting.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {

TWEAK_TEST(ExpandMacro);

TEST_F(ExpandMacroTest, Test) {
  Header = R"cpp(
    // error-ok: not real c++, just token manipulation
    #define FOO 1 2 3
    #define FUNC(X) X+X+X
    #define EMPTY
    #define EMPTY_FN(X)
  )cpp";

  // Available on macro names, not available anywhere else.
  EXPECT_AVAILABLE("^F^O^O^ BAR ^F^O^O^");
  EXPECT_AVAILABLE("^F^U^N^C^(1)");
  EXPECT_UNAVAILABLE("^#^d^efine^ ^XY^Z 1 ^2 ^3^");
  EXPECT_UNAVAILABLE("FOO ^B^A^R^ FOO ^");
  EXPECT_UNAVAILABLE("FUNC(^1^)^");

  // Works as expected on object-like macros.
  EXPECT_EQ(apply("^FOO BAR FOO"), "1 2 3 BAR FOO");
  EXPECT_EQ(apply("FOO BAR ^FOO"), "FOO BAR 1 2 3");
  // And function-like macros.
  EXPECT_EQ(apply("F^UNC(2)"), "2 + 2 + 2");

  // Works on empty macros.
  EXPECT_EQ(apply("int a ^EMPTY;"), "int a ;");
  EXPECT_EQ(apply("int a ^EMPTY_FN(1 2 3);"), "int a ;");
  EXPECT_EQ(apply("int a = 123 ^EMPTY EMPTY_FN(1);"),
            "int a = 123  EMPTY_FN(1);");
  EXPECT_EQ(apply("int a = 123 ^EMPTY_FN(1) EMPTY;"), "int a = 123  EMPTY;");
  EXPECT_EQ(apply("int a = 123 EMPTY_FN(1) ^EMPTY;"),
            "int a = 123 EMPTY_FN(1) ;");
}

} // namespace
} // namespace clangd
} // namespace clang
