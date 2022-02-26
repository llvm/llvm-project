//===-- DumpRecordLayoutTests.cpp -------------------------------*- C++ -*-===//
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

TWEAK_TEST(DumpRecordLayout);

TEST_F(DumpRecordLayoutTest, Test) {
  EXPECT_AVAILABLE("^s^truct ^X ^{ int x; ^};");
  EXPECT_THAT("struct X { int ^a; };", Not(isAvailable()));
  EXPECT_THAT("struct ^X;", Not(isAvailable()));
  EXPECT_THAT("template <typename T> struct ^X { T t; };", Not(isAvailable()));
  EXPECT_THAT("enum ^X {};", Not(isAvailable()));

  EXPECT_THAT(apply("struct ^X { int x; int y; };"),
              AllOf(StartsWith("message:"), HasSubstr("0 |   int x")));
}

} // namespace
} // namespace clangd
} // namespace clang
