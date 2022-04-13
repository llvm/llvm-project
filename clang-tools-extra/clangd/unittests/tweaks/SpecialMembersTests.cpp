//===-- SpecialMembersTests.cpp -------------------------------------------===//
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

TWEAK_TEST(SpecialMembers);

TEST_F(SpecialMembersTest, Test) {
  EXPECT_AVAILABLE("struct ^S {};");
  EXPECT_UNAVAILABLE("struct S { ^ };");
  EXPECT_UNAVAILABLE("union ^U {};");
  EXPECT_AVAILABLE("struct ^S { S(const S&); S(S&&); };");
  EXPECT_UNAVAILABLE("struct ^S {"
                     "S(const S&); S(S&&);"
                     "S &operator=(S&&); S &operator=(const S&);"
                     "};");

  const char *Output = R"cpp(struct S{S(const S &) = default;
  S(S &&) = default;
  S &operator=(const S &) = default;
  S &operator=(S &&) = default;
};)cpp";
  EXPECT_EQ(apply("struct ^S{};"), Output);

  Output = R"cpp(struct S{S(const S &) = default;
S(S &&) = default;
S &operator=(const S &) = delete;
S &operator=(S &&) = delete;
int& ref;};)cpp";
  EXPECT_EQ(apply("struct ^S{int& ref;};"), Output);
}

} // namespace
} // namespace clangd
} // namespace clang
