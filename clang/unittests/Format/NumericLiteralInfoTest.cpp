//===- unittest/Format/NumericLiteralInfoTest.cpp -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../../lib/Format/NumericLiteralInfo.h"
#include "gtest/gtest.h"

namespace clang {
namespace format {
namespace {

class NumericLiteralInfoTest : public testing::Test {
protected:
  NumericLiteralInfo getInfo(llvm::StringRef Text, char Separator = '\'') {
    return NumericLiteralInfo(Text, Separator);
  }
};

static constexpr auto npos = llvm::StringRef::npos;

TEST_F(NumericLiteralInfoTest, IntegerLiteral) {
  // Decimal.
  EXPECT_EQ(getInfo("90"), NumericLiteralInfo());
  EXPECT_EQ(getInfo("9L"), NumericLiteralInfo(npos, npos, npos, 1));
  EXPECT_EQ(getInfo("9'0U"), NumericLiteralInfo(npos, npos, npos, 3));

  // Octal.
  EXPECT_EQ(getInfo("07"), NumericLiteralInfo());
  EXPECT_EQ(getInfo("0z"), NumericLiteralInfo(npos, npos, npos, 1));
  // JavaScript.
  EXPECT_EQ(getInfo("0o7"), NumericLiteralInfo(1));
  EXPECT_EQ(getInfo("0O7_0", '_'), NumericLiteralInfo(1));

  // Binary.
  EXPECT_EQ(getInfo("0b1"), NumericLiteralInfo(1));
  EXPECT_EQ(getInfo("0B1ul"), NumericLiteralInfo(1, npos, npos, 3));

  // Hexadecimal.
  EXPECT_EQ(getInfo("0xF"), NumericLiteralInfo(1));
  EXPECT_EQ(getInfo("0XfZ"), NumericLiteralInfo(1, npos, npos, 3));
}

TEST_F(NumericLiteralInfoTest, FloatingPointLiteral) {
  // Decimal.
  EXPECT_EQ(getInfo(".9"), NumericLiteralInfo(npos, 0));
  EXPECT_EQ(getInfo("9."), NumericLiteralInfo(npos, 1));
  EXPECT_EQ(getInfo("9.F"), NumericLiteralInfo(npos, 1, npos, 2));
  EXPECT_EQ(getInfo("9e9"), NumericLiteralInfo(npos, npos, 1));
  EXPECT_EQ(getInfo("9E-9f"), NumericLiteralInfo(npos, npos, 1, 4));
  EXPECT_EQ(getInfo("9.9e+9bf16"), NumericLiteralInfo(npos, 1, 3, 6));

  // Hexadecimal.
  EXPECT_EQ(getInfo("0X.Fp9"), NumericLiteralInfo(1, 2, 4));
  EXPECT_EQ(getInfo("0xF.P9"), NumericLiteralInfo(1, 3, 4));
  EXPECT_EQ(getInfo("0xFp9"), NumericLiteralInfo(1, npos, 3));
  EXPECT_EQ(getInfo("0xFp+9F128"), NumericLiteralInfo(1, npos, 3, 6));
  EXPECT_EQ(getInfo("0xF.Fp-9_Pa"), NumericLiteralInfo(1, 3, 5, 8));
}

} // namespace
} // namespace format
} // namespace clang
