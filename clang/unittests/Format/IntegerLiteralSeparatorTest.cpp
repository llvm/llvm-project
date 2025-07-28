//===- unittest/Format/IntegerLiteralSeparatorTest.cpp --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FormatTestBase.h"

#define DEBUG_TYPE "integer-literal-separator-test"

namespace clang {
namespace format {
namespace test {
namespace {

class IntegerLiteralSeparatorTest : public FormatTestBase {};

TEST_F(IntegerLiteralSeparatorTest, SingleQuoteAsSeparator) {
  FormatStyle Style = getLLVMStyle();
  EXPECT_EQ(Style.Language, FormatStyle::LK_Cpp);
  EXPECT_EQ(Style.IntegerLiteralSeparator.Binary, 0);
  EXPECT_EQ(Style.IntegerLiteralSeparator.Decimal, 0);
  EXPECT_EQ(Style.IntegerLiteralSeparator.Hex, 0);

  constexpr StringRef Binary("b = 0b10011'11'0110'1u;");
  verifyFormat(Binary, Style);
  Style.IntegerLiteralSeparator.Binary = -1;
  verifyFormat("b = 0b100111101101u;", Binary, Style);
  Style.IntegerLiteralSeparator.Binary = 1;
  verifyFormat("b = 0b1'0'0'1'1'1'1'0'1'1'0'1u;", Binary, Style);
  Style.IntegerLiteralSeparator.Binary = 4;
  verifyFormat("b = 0b1001'1110'1101u;", Binary, Style);

  constexpr StringRef Decimal("d = 184467'440737'0'95505'92Ull;");
  verifyFormat(Decimal, Style);
  Style.IntegerLiteralSeparator.Decimal = -1;
  verifyFormat("d = 18446744073709550592Ull;", Decimal, Style);
  Style.IntegerLiteralSeparator.Decimal = 3;
  verifyFormat("d = 18'446'744'073'709'550'592Ull;", Decimal, Style);

  constexpr StringRef Hex("h = 0xDEAD'BEEF'DE'AD'BEE'Fuz;");
  verifyFormat(Hex, Style);
  Style.IntegerLiteralSeparator.Hex = -1;
  verifyFormat("h = 0xDEADBEEFDEADBEEFuz;", Hex, Style);
  Style.IntegerLiteralSeparator.Hex = 2;
  verifyFormat("h = 0xDE'AD'BE'EF'DE'AD'BE'EFuz;", Hex, Style);

  verifyFormat("o0 = 0;\n"
               "o1 = 07;\n"
               "o5 = 012345;",
               Style);

  verifyFormat("bi = 0b1'0000i;\n"
               "dif = 1'234if;\n"
               "hil = 0xA'BCil;",
               "bi = 0b10000i;\n"
               "dif = 1234if;\n"
               "hil = 0xABCil;",
               Style);

  verifyFormat("bd = 0b1'0000d;\n"
               "dh = 1'234h;\n"
               "dmin = 1'234min;\n"
               "dns = 1'234ns;\n"
               "ds = 1'234s;\n"
               "dus = 1'234us;\n"
               "hy = 0xA'BCy;",
               "bd = 0b10000d;\n"
               "dh = 1234h;\n"
               "dmin = 1234min;\n"
               "dns = 1234ns;\n"
               "ds = 1234s;\n"
               "dus = 1234us;\n"
               "hy = 0xABCy;",
               Style);

  verifyFormat("hd = 0xAB'Cd;", "hd = 0xABCd;", Style);

  verifyFormat("d = 5'678_km;\n"
               "h = 0xD'EF_u16;",
               "d = 5678_km;\n"
               "h = 0xDEF_u16;",
               Style);
}

TEST_F(IntegerLiteralSeparatorTest, UnderscoreAsSeparator) {
  FormatStyle Style = getLLVMStyle();
  constexpr StringRef Binary("B = 0B10011_11_0110_1;");
  constexpr StringRef Decimal("d = 184467_440737_0_95505_92;");
  constexpr StringRef Hex("H = 0XDEAD_BEEF_DE_AD_BEE_F;");

  auto TestUnderscore = [&](auto Language) {
    Style.Language = Language;

    Style.IntegerLiteralSeparator.Binary = 0;
    verifyFormat(Binary, Style);
    Style.IntegerLiteralSeparator.Binary = -1;
    verifyFormat("B = 0B100111101101;", Binary, Style);
    Style.IntegerLiteralSeparator.Binary = 4;
    verifyFormat("B = 0B1001_1110_1101;", Binary, Style);

    Style.IntegerLiteralSeparator.Decimal = 0;
    verifyFormat(Decimal, Style);
    Style.IntegerLiteralSeparator.Decimal = -1;
    verifyFormat("d = 18446744073709550592;", Decimal, Style);
    Style.IntegerLiteralSeparator.Decimal = 3;
    verifyFormat("d = 18_446_744_073_709_550_592;", Decimal, Style);

    Style.IntegerLiteralSeparator.Hex = 0;
    verifyFormat(Hex, Style);
    Style.IntegerLiteralSeparator.Hex = -1;
    verifyFormat("H = 0XDEADBEEFDEADBEEF;", Hex, Style);
    Style.IntegerLiteralSeparator.Hex = 2;
    verifyFormat("H = 0XDE_AD_BE_EF_DE_AD_BE_EF;", Hex, Style);
  };

  TestUnderscore(FormatStyle::LK_CSharp);
  TestUnderscore(FormatStyle::LK_Java);
  TestUnderscore(FormatStyle::LK_JavaScript);

  verifyFormat("d = 9_007_199_254_740_995n;", Style);
  verifyFormat("d = 9_007_199_254_740_995n;", "d = 9007199254740995n;", Style);

  Style.IntegerLiteralSeparator.Binary = 8;
  verifyFormat(
      "b = 0b100000_00000000_00000000_00000000_00000000_00000000_00000011n;",
      "b = 0b100000000000000000000000000000000000000000000000000011n;", Style);

  verifyFormat("h = 0x20_00_00_00_00_00_03n;", Style);
  verifyFormat("h = 0x20_00_00_00_00_00_03n;", "h = 0x20000000000003n;", Style);

  verifyFormat("o = 0o400000000000000003n;", Style);
}

TEST_F(IntegerLiteralSeparatorTest, MinDigits) {
  FormatStyle Style = getLLVMStyle();
  Style.IntegerLiteralSeparator.Binary = 3;
  Style.IntegerLiteralSeparator.Decimal = 3;
  Style.IntegerLiteralSeparator.Hex = 2;

  Style.IntegerLiteralSeparator.BinaryMinDigits = 7;
  verifyFormat("b1 = 0b101101;\n"
               "b2 = 0b1'101'101;",
               "b1 = 0b101'101;\n"
               "b2 = 0b1101101;",
               Style);

  Style.IntegerLiteralSeparator.DecimalMinDigits = 5;
  verifyFormat("d1 = 2023;\n"
               "d2 = 10'000;",
               "d1 = 2'023;\n"
               "d2 = 100'00;",
               Style);

  Style.IntegerLiteralSeparator.DecimalMinDigits = 3;
  verifyFormat("d1 = 123;\n"
               "d2 = 1'234;",
               "d1 = 12'3;\n"
               "d2 = 12'34;",
               Style);

  Style.IntegerLiteralSeparator.HexMinDigits = 6;
  verifyFormat("h1 = 0xABCDE;\n"
               "h2 = 0xAB'CD'EF;",
               "h1 = 0xA'BC'DE;\n"
               "h2 = 0xABC'DEF;",
               Style);
}

TEST_F(IntegerLiteralSeparatorTest, FixRanges) {
  FormatStyle Style = getLLVMStyle();
  Style.IntegerLiteralSeparator.Decimal = 3;

  constexpr StringRef Code("i = -12'34;\n"
                           "// clang-format off\n"
                           "j = 123'4;\n"
                           "// clang-format on\n"
                           "k = +1'23'4;");
  constexpr StringRef Expected("i = -1'234;\n"
                               "// clang-format off\n"
                               "j = 123'4;\n"
                               "// clang-format on\n"
                               "k = +1'234;");

  verifyFormat(Expected, Code, Style);

  verifyFormat("i = -1'234;\n"
               "// clang-format off\n"
               "j = 123'4;\n"
               "// clang-format on\n"
               "k = +1'23'4;",
               Code, Style, {tooling::Range(0, 11)}); // line 1

  verifyFormat(Code, Code, Style, {tooling::Range(32, 10)}); // line 3

  verifyFormat("i = -12'34;\n"
               "// clang-format off\n"
               "j = 123'4;\n"
               "// clang-format on\n"
               "k = +1'234;",
               Code, Style, {tooling::Range(61, 12)}); // line 5

  verifyFormat(Expected, Code, Style,
               {tooling::Range(0, 11), tooling::Range(61, 12)}); // lines 1, 5
}

TEST_F(IntegerLiteralSeparatorTest, FloatingPoint) {
  FormatStyle Style = getLLVMStyle();
  Style.IntegerLiteralSeparator.Decimal = 3;
  Style.IntegerLiteralSeparator.Hex = 2;

  verifyFormat("d0 = .0;\n"
               "d1 = 0.;\n"
               "y = 7890.;\n"
               "E = 3456E2;\n"
               "p = 0xABCp2;",
               Style);

  Style.Language = FormatStyle::LK_JavaScript;
  verifyFormat("y = 7890.;\n"
               "e = 3456e2;",
               Style);

  Style.Language = FormatStyle::LK_Java;
  verifyFormat("y = 7890.;\n"
               "E = 3456E2;\n"
               "P = 0xABCP2;\n"
               "f = 1234f;\n"
               "D = 5678D;",
               Style);

  Style.Language = FormatStyle::LK_CSharp;
  verifyFormat("y = 7890.;\n"
               "e = 3456e2;\n"
               "F = 1234F;\n"
               "d = 5678d;\n"
               "M = 9012M",
               Style);
}

} // namespace
} // namespace test
} // namespace format
} // namespace clang
