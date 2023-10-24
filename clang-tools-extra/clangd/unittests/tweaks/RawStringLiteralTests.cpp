//===-- RawStringLiteralTests.cpp -------------------------------*- C++ -*-===//
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

TWEAK_TEST(RawStringLiteral);

TEST_F(RawStringLiteralTest, Test) {
  Context = Expression;
  EXPECT_AVAILABLE(R"cpp(^"^f^o^o^\^n^")cpp");
  EXPECT_AVAILABLE(R"cpp(R"(multi )" ^"token " "str\ning")cpp");
  EXPECT_UNAVAILABLE(R"cpp(^"f^o^o^o")cpp"); // no chars need escaping
  EXPECT_UNAVAILABLE(R"cpp(R"(multi )" ^"token " u8"str\ning")cpp"); // nonascii
  EXPECT_UNAVAILABLE(R"cpp(^R^"^(^multi )" "token " "str\ning")cpp"); // raw
  EXPECT_UNAVAILABLE(R"cpp(^"token\n" __FILE__)cpp"); // chunk is macro
  EXPECT_UNAVAILABLE(R"cpp(^"a\r\n";)cpp");           // forbidden escape char

  const char *Input = R"cpp(R"(multi
token)" "\nst^ring\n" "literal")cpp";
  const char *Output = R"cpp(R"(multi
token
string
literal)")cpp";
  EXPECT_EQ(apply(Input), Output);
}

TEST_F(RawStringLiteralTest, TestC) {
  Context = File;
  FileName = "TestTU.c";
  ExtraArgs = {"-xc"}; // raw strings are unavailable in C
  EXPECT_UNAVAILABLE(R"c(const char *a = ^"^f^o^o^\^n^";)c");
}

TEST_F(RawStringLiteralTest, TestCpp98) {
  Context = File;
  ExtraArgs = {"-std=c++98"}; // raw strings are unavailable
                              // in versions prior to C++11
  EXPECT_UNAVAILABLE(R"cpp(const char *a = ^"^f^o^o^\^n^";)cpp");
}

} // namespace
} // namespace clangd
} // namespace clang
