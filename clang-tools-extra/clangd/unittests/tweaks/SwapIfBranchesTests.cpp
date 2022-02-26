//===-- SwapIfBranchesTests.cpp ---------------------------------*- C++ -*-===//
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

TWEAK_TEST(SwapIfBranches);

TEST_F(SwapIfBranchesTest, Test) {
  Context = Function;
  EXPECT_EQ(apply("^if (true) {return;} else {(void)0;}"),
            "if (true) {(void)0;} else {return;}");
  EXPECT_EQ(apply("^if (/*error-ok*/) {return;} else {(void)0;}"),
            "if (/*error-ok*/) {(void)0;} else {return;}")
      << "broken condition";
  EXPECT_AVAILABLE("^i^f^^(^t^r^u^e^) { return; } ^e^l^s^e^ { return; }");
  EXPECT_UNAVAILABLE("if (true) {^return ^;^ } else { ^return^;^ }");
  // Available in subexpressions of the condition;
  EXPECT_THAT("if(2 + [[2]] + 2) { return; } else {return;}", isAvailable());
  // But not as part of the branches.
  EXPECT_THAT("if(2 + 2 + 2) { [[return]]; } else { return; }",
              Not(isAvailable()));
  // Range covers the "else" token, so available.
  EXPECT_THAT("if(2 + 2 + 2) { return[[; } else {return;]]}", isAvailable());
  // Not available in compound statements in condition.
  EXPECT_THAT("if([]{return [[true]];}()) { return; } else { return; }",
              Not(isAvailable()));
  // Not available if both sides aren't braced.
  EXPECT_THAT("^if (1) return; else { return; }", Not(isAvailable()));
  // Only one if statement is supported!
  EXPECT_THAT("[[if(1){}else{}if(2){}else{}]]", Not(isAvailable()));
}

} // namespace
} // namespace clangd
} // namespace clang
