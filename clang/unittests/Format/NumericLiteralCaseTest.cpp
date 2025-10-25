//===- unittest/Format/NumericLiteralCaseTest.cpp -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FormatTestBase.h"

#define DEBUG_TYPE "numeric-literal-case-test"

namespace clang {
namespace format {
namespace test {
namespace {

class NumericLiteralCaseTest : public FormatTestBase {};

TEST_F(NumericLiteralCaseTest, Prefix) {
  constexpr StringRef Bin0("b = 0b0'10'010uL;");
  constexpr StringRef Bin1("b = 0B010'010Ul;");
  constexpr StringRef Hex0("b = 0xdead'BEEFuL;");
  constexpr StringRef Hex1("b = 0Xdead'BEEFUl;");
  verifyFormat(Bin0);
  verifyFormat(Bin1);
  verifyFormat(Hex0);
  verifyFormat(Hex1);

  auto Style = getLLVMStyle();
  EXPECT_EQ(Style.NumericLiteralCase.Prefix, FormatStyle::NLCS_Leave);
  EXPECT_EQ(Style.NumericLiteralCase.HexDigit, FormatStyle::NLCS_Leave);
  EXPECT_EQ(Style.NumericLiteralCase.ExponentLetter, FormatStyle::NLCS_Leave);
  EXPECT_EQ(Style.NumericLiteralCase.Suffix, FormatStyle::NLCS_Leave);

  Style.NumericLiteralCase.Prefix = FormatStyle::NLCS_Upper;
  verifyFormat("b = 0B0'10'010uL;", Bin0, Style);
  verifyFormat(Bin1, Style);
  verifyFormat("b = 0Xdead'BEEFuL;", Hex0, Style);
  verifyFormat(Hex1, Style);
  verifyFormat("i = 0XaBcD.a0Ebp123F;", Style);
  verifyFormat("j = 0XaBcD.a0EbP123f;", Style);

  Style.NumericLiteralCase.Prefix = FormatStyle::NLCS_Lower;
  verifyFormat(Bin0, Style);
  verifyFormat("b = 0b010'010Ul;", Bin1, Style);
  verifyFormat(Hex0, Style);
  verifyFormat("b = 0xdead'BEEFUl;", Hex1, Style);
}

TEST_F(NumericLiteralCaseTest, HexDigit) {
  constexpr StringRef A("a = 0xaBc0'123fuL;");
  constexpr StringRef B("b = 0XaBc0'123FUl;");
  constexpr StringRef C("c = 0xa'Bc.0p12'3f32;");
  constexpr StringRef D("d = 0xa'Bc.0P12'3F128;");
  constexpr StringRef E("e = 0b0011'00Ull;");
  constexpr StringRef F("f = 0B0100'000zu;");
  constexpr StringRef G("g = 0.123e-19f;");
  constexpr StringRef H("h = 0.12'3E-19F16;");
  constexpr StringRef I("i = 0x.0000aBcp12'3F128;");
  constexpr StringRef J("j = 0xaa1'fP12'3F128;");
  constexpr StringRef K("k = 0x0;");
  constexpr StringRef L("l = 0xA;");
  verifyFormat(A);
  verifyFormat(B);
  verifyFormat(C);
  verifyFormat(D);
  verifyFormat(E);
  verifyFormat(F);
  verifyFormat(G);
  verifyFormat(H);
  verifyFormat(I);
  verifyFormat(J);
  verifyFormat(K);
  verifyFormat(L);

  auto Style = getLLVMStyle();
  Style.NumericLiteralCase.HexDigit = FormatStyle::NLCS_Upper;
  verifyFormat("a = 0xABC0'123FuL;", A, Style);
  verifyFormat("b = 0XABC0'123FUl;", B, Style);
  verifyFormat("c = 0xA'BC.0p12'3f32;", C, Style);
  verifyFormat("d = 0xA'BC.0P12'3F128;", D, Style);
  verifyFormat(E, Style);
  verifyFormat(F, Style);
  verifyFormat(G, Style);
  verifyFormat(H, Style);
  verifyFormat("i = 0x.0000ABCp12'3F128;", I, Style);
  verifyFormat("j = 0xAA1'FP12'3F128;", J, Style);
  verifyFormat(K, Style);
  verifyFormat(L, Style);

  Style.NumericLiteralCase.HexDigit = FormatStyle::NLCS_Lower;
  verifyFormat("a = 0xabc0'123fuL;", A, Style);
  verifyFormat("b = 0Xabc0'123fUl;", B, Style);
  verifyFormat("c = 0xa'bc.0p12'3f32;", C, Style);
  verifyFormat("d = 0xa'bc.0P12'3F128;", D, Style);
  verifyFormat(E, Style);
  verifyFormat(F, Style);
  verifyFormat(G, Style);
  verifyFormat(H, Style);
  verifyFormat("i = 0x.0000abcp12'3F128;", I, Style);
  verifyFormat("j = 0xaa1'fP12'3F128;", J, Style);
  verifyFormat(K, Style);
  verifyFormat("l = 0xa;", Style);
}

TEST_F(NumericLiteralCaseTest, ExponentLetter) {
  constexpr StringRef A("a = .0'01e-19f;");
  constexpr StringRef B("b = .00'1E2F;");
  constexpr StringRef C("c = 10'2.e99;");
  constexpr StringRef D("d = 123.456E-1;");
  constexpr StringRef E("e = 0x12abEe3.456p-10'0;");
  constexpr StringRef F("f = 0x.deEfP23;");
  constexpr StringRef G("g = 0xe0E1.p-1;");
  verifyFormat(A);
  verifyFormat(B);
  verifyFormat(C);
  verifyFormat(D);
  verifyFormat(E);
  verifyFormat(F);
  verifyFormat(G);

  auto Style = getLLVMStyle();
  Style.NumericLiteralCase.ExponentLetter = FormatStyle::NLCS_Lower;
  verifyFormat(A, Style);
  verifyFormat("b = .00'1e2F;", B, Style);
  verifyFormat(C, Style);
  verifyFormat("d = 123.456e-1;", D, Style);
  verifyFormat(E, Style);
  verifyFormat("f = 0x.deEfp23;", F, Style);
  verifyFormat(G, Style);

  Style.NumericLiteralCase.ExponentLetter = FormatStyle::NLCS_Upper;
  verifyFormat("a = .0'01E-19f;", A, Style);
  verifyFormat(B, Style);
  verifyFormat("c = 10'2.E99;", C, Style);
  verifyFormat(D, Style);
  verifyFormat("e = 0x12abEe3.456P-10'0;", E, Style);
  verifyFormat(F, Style);
  verifyFormat("g = 0xe0E1.P-1;", G, Style);
}

TEST_F(NumericLiteralCaseTest, IntegerSuffix) {
  constexpr StringRef A("a = 102u;");
  constexpr StringRef B("b = 0177U;");
  constexpr StringRef C("c = 0b101'111llU;");
  constexpr StringRef D("d = 0xdead'BeefuZ;");
  constexpr StringRef E("e = 3lU;");
  constexpr StringRef F("f = 1zu;");
  constexpr StringRef G("g = 0uLL;");
  constexpr StringRef H("h = 10'233'213'0101uLL;");
  verifyFormat(A);
  verifyFormat(B);
  verifyFormat(C);
  verifyFormat(D);
  verifyFormat(E);
  verifyFormat(F);
  verifyFormat(G);
  verifyFormat(H);

  auto Style = getLLVMStyle();
  Style.NumericLiteralCase.Suffix = FormatStyle::NLCS_Lower;
  verifyFormat(A, Style);
  verifyFormat("b = 0177u;", B, Style);
  verifyFormat("c = 0b101'111llu;", C, Style);
  verifyFormat("d = 0xdead'Beefuz;", D, Style);
  verifyFormat("e = 3lu;", E, Style);
  verifyFormat(F, Style);
  verifyFormat("g = 0ull;", G, Style);
  verifyFormat("h = 10'233'213'0101ull;", H, Style);

  Style.NumericLiteralCase.Suffix = FormatStyle::NLCS_Upper;
  verifyFormat("a = 102U;", A, Style);
  verifyFormat(B, Style);
  verifyFormat("c = 0b101'111LLU;", C, Style);
  verifyFormat("d = 0xdead'BeefUZ;", D, Style);
  verifyFormat("e = 3LU;", E, Style);
  verifyFormat("f = 1ZU;", F, Style);
  verifyFormat("g = 0ULL;", G, Style);
  verifyFormat("h = 10'233'213'0101ULL;", H, Style);
}

TEST_F(NumericLiteralCaseTest, FloatingPointSuffix) {
  auto Style = getLLVMStyle();
  // Floating point literals without suffixes.
  constexpr std::array<StringRef, 6> FloatingPointStatements = {
      "a = 0.",       "b = 1.0",        "c = .123'45E-10",
      "d = 12'3.0e1", "e = 0Xa0eE.P10", "f = 0xeE01.aFf3p6",
  };

  // All legal floating-point literal suffixes defined in the C++23 standard in
  // lowercase.
  constexpr std::array<StringRef, 7> FloatingPointSuffixes = {
      "f", "l", "f16", "f32", "f64", "f128", "bf16",
  };

  // Test all combinations of literals with suffixes.
  for (const auto &Statement : FloatingPointStatements) {
    for (const auto &Suffix : FloatingPointSuffixes) {
      const auto LowerLine = Statement.str() + Suffix.str() + ";";
      const auto UpperLine = Statement.str() + Suffix.upper() + ";";

      Style.NumericLiteralCase.Suffix = FormatStyle::NLCS_Leave;
      verifyFormat(LowerLine, Style);
      verifyFormat(UpperLine, Style);

      Style.NumericLiteralCase.Suffix = FormatStyle::NLCS_Lower;
      verifyFormat(LowerLine, Style);
      verifyFormat(LowerLine, UpperLine, Style);

      Style.NumericLiteralCase.Suffix = FormatStyle::NLCS_Upper;
      verifyFormat(UpperLine, LowerLine, Style);
      verifyFormat(UpperLine, Style);
    }
  }
}

TEST_F(NumericLiteralCaseTest, CppStandardAndUserDefinedLiteralsAreUntouched) {
  auto Style = getLLVMStyle();
  Style.NumericLiteralCase.Prefix = FormatStyle::NLCS_Upper;
  Style.NumericLiteralCase.HexDigit = FormatStyle::NLCS_Upper;
  Style.NumericLiteralCase.ExponentLetter = FormatStyle::NLCS_Upper;
  Style.NumericLiteralCase.Suffix = FormatStyle::NLCS_Upper;

  // C++ user-defined suffixes begin with '_' or are reserved for the standard
  // library.
  constexpr StringRef UDLiterals("a = 12.if;\n"
                                 "b = -3i;\n"
                                 "c = 100'01il;\n"
                                 "d = 100'0.12il;\n"
                                 "e = 12h;\n"
                                 "f = 0XABE12h;\n"
                                 "g = 0XFA03min;\n"
                                 "h = 0X12B4Ds;\n"
                                 "i = 20.13E-1ms;\n"
                                 "j = 20.13E-1us;\n"
                                 "k = 20.13E-1ns;\n"
                                 "l = 20.13E-1y;\n"
                                 "m = 20.13E-1d;\n"
                                 "n = 20.13E-1d;\n"
                                 "o = 1d;\n"
                                 "p = 102_ffl_lzlz;\n"
                                 "q = 10.2_l;\n"
                                 "r = 0XABDE.0'1P-23_f;\n"
                                 "s = 102_foo_bar;\n"
                                 "t = 123.456_felfz_ballpen;\n"
                                 "u = 0XBEAD1_spacebar;");

  verifyFormat(UDLiterals, Style);
  Style.NumericLiteralCase.Suffix = FormatStyle::NLCS_Lower;
  verifyFormat(UDLiterals, Style);
}

TEST_F(NumericLiteralCaseTest, FixRanges) {
  auto Style = getLLVMStyle();
  Style.NumericLiteralCase.Prefix = FormatStyle::NLCS_Lower;
  Style.NumericLiteralCase.HexDigit = FormatStyle::NLCS_Lower;
  Style.NumericLiteralCase.ExponentLetter = FormatStyle::NLCS_Lower;
  Style.NumericLiteralCase.Suffix = FormatStyle::NLCS_Lower;

  constexpr StringRef CodeBlock("a = 0xFea3duLL;\n"
                                "b = 0X.aEbp-12f;\n"
                                "c = 0uLL;\n"
                                "// clang-format off\n"
                                "e = 0xBeAdu;\n"
                                "// clang-format on\n"
                                "g = 0xabCDu;\n"
                                "h = 0b010uL;\n"
                                "// clang-format off\n"
                                "i = 0B1010'000Zu;\n"
                                "// clang-format on\n"
                                "k = 0XaBuL;");

  verifyFormat("a = 0xfea3dull;\n"
               "b = 0x.aebp-12f;\n"
               "c = 0ull;\n"
               "// clang-format off\n"
               "e = 0xBeAdu;\n"
               "// clang-format on\n"
               "g = 0xabcdu;\n"
               "h = 0b010ul;\n"
               "// clang-format off\n"
               "i = 0B1010'000Zu;\n"
               "// clang-format on\n"
               "k = 0xabul;",
               CodeBlock, Style);
}

TEST_F(NumericLiteralCaseTest, UnderScoreSeparatorLanguages) {
  auto Style = getLLVMStyle();

  constexpr StringRef CodeBlock("a = 0xFea_3dl;\n"
                                "b = 0123_345;\n"
                                "c = 0b11____00lU;\n"
                                "d = 0XB_e_A_du;\n"
                                "e = 123_456.333__456e-10f;\n"
                                "f = .1_0E-10D;\n"
                                "g = 1_0.F;\n"
                                "h = 0B1_0;");
  auto TestUnderscore = [&](auto Language) {
    Style.Language = Language;
    Style.NumericLiteralCase.Prefix = FormatStyle::NLCS_Lower;
    Style.NumericLiteralCase.HexDigit = FormatStyle::NLCS_Upper;
    Style.NumericLiteralCase.ExponentLetter = FormatStyle::NLCS_Lower;
    Style.NumericLiteralCase.Suffix = FormatStyle::NLCS_Upper;
    verifyFormat("a = 0xFEA_3DL;\n"
                 "b = 0123_345;\n"
                 "c = 0b11____00LU;\n"
                 "d = 0xB_E_A_DU;\n"
                 "e = 123_456.333__456e-10F;\n"
                 "f = .1_0e-10D;\n"
                 "g = 1_0.F;\n"
                 "h = 0b1_0;",
                 CodeBlock, Style);

    Style.NumericLiteralCase.Prefix = FormatStyle::NLCS_Upper;
    Style.NumericLiteralCase.HexDigit = FormatStyle::NLCS_Lower;
    Style.NumericLiteralCase.ExponentLetter = FormatStyle::NLCS_Upper;
    Style.NumericLiteralCase.Suffix = FormatStyle::NLCS_Lower;

    verifyFormat("a = 0Xfea_3dl;\n"
                 "b = 0123_345;\n"
                 "c = 0B11____00lu;\n"
                 "d = 0Xb_e_a_du;\n"
                 "e = 123_456.333__456E-10f;\n"
                 "f = .1_0E-10d;\n"
                 "g = 1_0.f;\n"
                 "h = 0B1_0;",
                 CodeBlock, Style);
  };

  TestUnderscore(FormatStyle::LK_CSharp);
  TestUnderscore(FormatStyle::LK_Java);
  TestUnderscore(FormatStyle::LK_JavaScript);

  Style.Language = FormatStyle::LK_JavaScript;
  Style.NumericLiteralCase.Prefix = FormatStyle::NLCS_Upper;
  verifyFormat("o = 0O0_10_010;", "o = 0o0_10_010;", Style);
  Style.NumericLiteralCase.Prefix = FormatStyle::NLCS_Lower;
  verifyFormat("o = 0o0_10_010;", "o = 0O0_10_010;", Style);
}

} // namespace
} // namespace test
} // namespace format
} // namespace clang
