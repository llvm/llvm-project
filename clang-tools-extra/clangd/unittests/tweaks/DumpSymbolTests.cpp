//===-- DumpSymbolTests.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TweakTesting.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <string>

using ::testing::AllOf;
using ::testing::HasSubstr;
using ::testing::StartsWith;

namespace clang {
namespace clangd {
namespace {

TWEAK_TEST(DumpSymbol);

TEST_F(DumpSymbolTest, Test) {
  std::string ID = R"("id":"CA2EBE44A1D76D2A")";
  std::string USR = R"("usr":"c:@F@foo#")";
  EXPECT_THAT(apply("void f^oo();"),
              AllOf(StartsWith("message:"), testing::HasSubstr(ID),
                    testing::HasSubstr(USR)));
}

} // namespace
} // namespace clangd
} // namespace clang
