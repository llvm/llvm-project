//===- unittest/Format/IntegerLiteralSeparatorTest.cpp --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Format/Format.h"

#include "../Tooling/ReplacementTest.h"
#include "FormatTestUtils.h"

#define DEBUG_TYPE "integer-literal-separator-test"

namespace clang {
namespace format {
namespace {

// TODO:
// Refactor the class declaration, which is copied from BracesInserterTest.cpp.
class IntegerLiteralSeparatorTest : public ::testing::Test {
protected:
  std::string format(llvm::StringRef Code, const FormatStyle &Style,
                     const std::vector<tooling::Range> &Ranges) {
    LLVM_DEBUG(llvm::errs() << "---\n");
    LLVM_DEBUG(llvm::errs() << Code << "\n\n");
    auto NonEmptyRanges = Ranges;
    if (Ranges.empty())
      NonEmptyRanges = {1, tooling::Range(0, Code.size())};
    FormattingAttemptStatus Status;
    tooling::Replacements Replaces =
        reformat(Style, Code, NonEmptyRanges, "<stdin>", &Status);
    EXPECT_EQ(true, Status.FormatComplete) << Code << "\n\n";
    ReplacementCount = Replaces.size();
    auto Result = applyAllReplacements(Code, Replaces);
    EXPECT_TRUE(static_cast<bool>(Result));
    LLVM_DEBUG(llvm::errs() << "\n" << *Result << "\n\n");
    return *Result;
  }

  void _verifyFormat(const char *File, int Line, llvm::StringRef Expected,
                     llvm::StringRef Code,
                     const FormatStyle &Style = getLLVMStyle(),
                     const std::vector<tooling::Range> &Ranges = {}) {
    testing::ScopedTrace t(File, Line, ::testing::Message() << Code.str());
    EXPECT_EQ(Expected.str(), format(Expected, Style, Ranges))
        << "Expected code is not stable";
    EXPECT_EQ(Expected.str(), format(Code, Style, Ranges));
    if (Style.Language == FormatStyle::LK_Cpp && Ranges.empty()) {
      // Objective-C++ is a superset of C++, so everything checked for C++
      // needs to be checked for Objective-C++ as well.
      FormatStyle ObjCStyle = Style;
      ObjCStyle.Language = FormatStyle::LK_ObjC;
      EXPECT_EQ(Expected.str(), format(test::messUp(Code), ObjCStyle, Ranges));
    }
  }

  void _verifyFormat(const char *File, int Line, llvm::StringRef Code,
                     const FormatStyle &Style = getLLVMStyle(),
                     const std::vector<tooling::Range> &Ranges = {}) {
    _verifyFormat(File, Line, Code, Code, Style, Ranges);
  }

  int ReplacementCount;
};

#define verifyFormat(...) _verifyFormat(__FILE__, __LINE__, __VA_ARGS__)

TEST_F(IntegerLiteralSeparatorTest, SingleQuoteAsSeparator) {
  FormatStyle Style = getLLVMStyle();
  EXPECT_EQ(Style.Language, FormatStyle::LK_Cpp);
  EXPECT_EQ(Style.IntegerLiteralSeparator.Binary, 0);
  EXPECT_EQ(Style.IntegerLiteralSeparator.Decimal, 0);
  EXPECT_EQ(Style.IntegerLiteralSeparator.Hex, 0);

  const StringRef Binary("b = 0b10011'11'0110'1u;");
  verifyFormat(Binary, Style);
  Style.IntegerLiteralSeparator.Binary = -1;
  verifyFormat("b = 0b100111101101u;", Binary, Style);
  Style.IntegerLiteralSeparator.Binary = 1;
  verifyFormat("b = 0b1'0'0'1'1'1'1'0'1'1'0'1u;", Binary, Style);
  Style.IntegerLiteralSeparator.Binary = 4;
  verifyFormat("b = 0b1001'1110'1101u;", Binary, Style);

  const StringRef Decimal("d = 184467'440737'0'95505'92Ull;");
  verifyFormat(Decimal, Style);
  Style.IntegerLiteralSeparator.Decimal = -1;
  verifyFormat("d = 18446744073709550592Ull;", Decimal, Style);
  Style.IntegerLiteralSeparator.Decimal = 3;
  verifyFormat("d = 18'446'744'073'709'550'592Ull;", Decimal, Style);

  const StringRef Hex("h = 0xDEAD'BEEF'DE'AD'BEE'Fuz;");
  verifyFormat(Hex, Style);
  Style.IntegerLiteralSeparator.Hex = -1;
  verifyFormat("h = 0xDEADBEEFDEADBEEFuz;", Hex, Style);
  Style.IntegerLiteralSeparator.Hex = 2;
  verifyFormat("h = 0xDE'AD'BE'EF'DE'AD'BE'EFuz;", Hex, Style);

  verifyFormat("o0 = 0;\n"
               "o1 = 07;\n"
               "o5 = 012345",
               Style);
}

TEST_F(IntegerLiteralSeparatorTest, UnderscoreAsSeparator) {
  FormatStyle Style = getLLVMStyle();
  const StringRef Binary("B = 0B10011_11_0110_1;");
  const StringRef Decimal("d = 184467_440737_0_95505_92;");
  const StringRef Hex("H = 0XDEAD_BEEF_DE_AD_BEE_F;");

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

TEST_F(IntegerLiteralSeparatorTest, FixRanges) {
  FormatStyle Style = getLLVMStyle();
  Style.IntegerLiteralSeparator.Decimal = 3;

  const StringRef Code("i = -12'34;\n"
                       "// clang-format off\n"
                       "j = 123'4;\n"
                       "// clang-format on\n"
                       "k = +1'23'4;");
  const StringRef Expected("i = -1'234;\n"
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

  verifyFormat(Code, Style, {tooling::Range(32, 10)}); // line 3

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
} // namespace format
} // namespace clang
