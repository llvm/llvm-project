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

static constexpr auto npos = llvm::StringRef::npos;

class NumericLiteralInfoTest : public testing::Test {
protected:
  bool verifyInfo(const NumericLiteralInfo &Info, size_t BaseLetterPos = npos,
                  size_t DotPos = npos, size_t ExponentLetterPos = npos,
                  size_t SuffixPos = npos) {
    return Info.BaseLetterPos == BaseLetterPos && Info.DotPos == DotPos &&
           Info.ExponentLetterPos == ExponentLetterPos &&
           Info.SuffixPos == SuffixPos;
  }
};

TEST_F(NumericLiteralInfoTest, IntegerLiteral) {
  // Decimal.
  EXPECT_TRUE(verifyInfo(NumericLiteralInfo("90")));
  EXPECT_TRUE(verifyInfo(NumericLiteralInfo("9L"), npos, npos, npos, 1));
  EXPECT_TRUE(verifyInfo(NumericLiteralInfo("9'0U"), npos, npos, npos, 3));

  // Octal.
  EXPECT_TRUE(verifyInfo(NumericLiteralInfo("0")));
  EXPECT_TRUE(verifyInfo(NumericLiteralInfo("07")));
  EXPECT_TRUE(verifyInfo(NumericLiteralInfo("0z"), npos, npos, npos, 1));
  // JavaScript.
  EXPECT_TRUE(verifyInfo(NumericLiteralInfo("0o7"), 1));
  EXPECT_TRUE(verifyInfo(NumericLiteralInfo("0O7_0", '_'), 1));

  // Binary.
  EXPECT_TRUE(verifyInfo(NumericLiteralInfo("0b1"), 1));
  EXPECT_TRUE(verifyInfo(NumericLiteralInfo("0B1ul"), 1, npos, npos, 3));

  // Hexadecimal.
  EXPECT_TRUE(verifyInfo(NumericLiteralInfo("0xF"), 1));
  EXPECT_TRUE(verifyInfo(NumericLiteralInfo("0XfZ"), 1, npos, npos, 3));
}

TEST_F(NumericLiteralInfoTest, FloatingPointLiteral) {
  // Decimal.
  EXPECT_TRUE(verifyInfo(NumericLiteralInfo(".9"), npos, 0));
  EXPECT_TRUE(verifyInfo(NumericLiteralInfo("9."), npos, 1));
  EXPECT_TRUE(verifyInfo(NumericLiteralInfo("9.F"), npos, 1, npos, 2));
  EXPECT_TRUE(verifyInfo(NumericLiteralInfo("9e9"), npos, npos, 1));
  EXPECT_TRUE(verifyInfo(NumericLiteralInfo("9E-9f"), npos, npos, 1, 4));
  EXPECT_TRUE(verifyInfo(NumericLiteralInfo("9.9e+9bf16"), npos, 1, 3, 6));

  // Hexadecimal.
  EXPECT_TRUE(verifyInfo(NumericLiteralInfo("0X.Fp9"), 1, 2, 4));
  EXPECT_TRUE(verifyInfo(NumericLiteralInfo("0xF.P9"), 1, 3, 4));
  EXPECT_TRUE(verifyInfo(NumericLiteralInfo("0xFp9"), 1, npos, 3));
  EXPECT_TRUE(verifyInfo(NumericLiteralInfo("0xFp+9F128"), 1, npos, 3, 6));
  EXPECT_TRUE(verifyInfo(NumericLiteralInfo("0xF.Fp-9_Pa"), 1, 3, 5, 8));
}

} // namespace
} // namespace format
} // namespace clang
