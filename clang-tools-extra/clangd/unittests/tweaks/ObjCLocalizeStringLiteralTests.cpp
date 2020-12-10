//===-- ObjCLocalizeStringLiteralTests.cpp ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestTU.h"
#include "TweakTesting.h"
#include "gmock/gmock-matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {

TWEAK_TEST(ObjCLocalizeStringLiteral);

TEST_F(ObjCLocalizeStringLiteralTest, Test) {
  ExtraArgs.push_back("-x");
  ExtraArgs.push_back("objective-c");

  // Ensure the action can be initiated in the string literal.
  EXPECT_AVAILABLE(R"(id x = ^[[@[[^"^t^est^"]]]];)");

  // Ensure that the action can't be initiated in other places.
  EXPECT_UNAVAILABLE(R"([[i^d ^[[x]] ^= @"test";^]])");

  // Ensure that the action is not available for regular C strings.
  EXPECT_UNAVAILABLE(R"(const char * x= "^test";)");

  const char *Input = R"(id x = [[@"test"]];)";
  const char *Output = R"(id x = NSLocalizedString(@"test", @"");)";
  EXPECT_EQ(apply(Input), Output);
}

} // namespace
} // namespace clangd
} // namespace clang
