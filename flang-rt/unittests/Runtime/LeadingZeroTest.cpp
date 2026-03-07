//===-- unittests/Runtime/LeadingZeroTest.cpp --------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for F202X leading-zero control edit descriptors: LZ, LZP, LZS.
// LZ  - processor-dependent (flang prints leading zero)
// LZP - print the optional leading zero
// LZS - suppress the optional leading zero
//
//===----------------------------------------------------------------------===//

#include "CrashHandlerFixture.h"
#include "flang-rt/runtime/descriptor.h"
#include "flang/Runtime/io-api.h"
#include <algorithm>
#include <cstring>
#include <gtest/gtest.h>
#include <string>
#include <tuple>
#include <vector>

using namespace Fortran::runtime;
using namespace Fortran::runtime::io;

static bool CompareFormattedStrings(
    const std::string &expect, const std::string &got) {
  std::string want{expect};
  want.resize(got.size(), ' ');
  return want == got;
}

// Perform format on a double and return the trimmed result
static std::string FormatReal(const char *format, double x) {
  char buffer[800];
  auto cookie{IONAME(BeginInternalFormattedOutput)(
      buffer, sizeof buffer, format, std::strlen(format))};
  EXPECT_TRUE(IONAME(OutputReal64)(cookie, x));
  auto status{IONAME(EndIoStatement)(cookie)};
  EXPECT_EQ(status, 0);
  std::string got{buffer, sizeof buffer};
  auto lastNonBlank{got.find_last_not_of(" ")};
  if (lastNonBlank != std::string::npos) {
    got.resize(lastNonBlank + 1);
  }
  return got;
}

static bool CompareFormatReal(
    const char *format, double x, const char *expect, std::string &got) {
  got = FormatReal(format, x);
  return CompareFormattedStrings(expect, got);
}

struct LeadingZeroTests : CrashHandlerFixture {};

// LZP with F editing: value < 1 should print "0." before decimal digits
TEST(LeadingZeroTests, LZP_F_editing) {
  static constexpr std::pair<const char *, const char *> cases[]{
      {"(LZP,F6.1)", "   0.2"},
      {"(LZP,F10.3)", "     0.200"},
      {"(LZP,F6.1)", "   0.5"},
      {"(LZP,F4.1)", " 0.1"},
  };
  double values[]{0.2, 0.2, 0.5, 0.1};
  for (int i = 0; i < 4; ++i) {
    std::string got;
    ASSERT_TRUE(
        CompareFormatReal(cases[i].first, values[i], cases[i].second, got))
        << "Failed: format=" << cases[i].first << " value=" << values[i]
        << ", expected '" << cases[i].second << "', got '" << got << "'";
  }
}

// LZS with F editing: value < 1 should suppress the leading zero
TEST(LeadingZeroTests, LZS_F_editing) {
  static constexpr std::pair<const char *, const char *> cases[]{
      {"(LZS,F6.1)", "    .2"},
      {"(LZS,F10.3)", "      .200"},
      {"(LZS,F6.1)", "    .5"},
      {"(LZS,F4.1)", "  .1"},
  };
  double values[]{0.2, 0.2, 0.5, 0.1};
  for (int i = 0; i < 4; ++i) {
    std::string got;
    ASSERT_TRUE(
        CompareFormatReal(cases[i].first, values[i], cases[i].second, got))
        << "Failed: format=" << cases[i].first << " value=" << values[i]
        << ", expected '" << cases[i].second << "', got '" << got << "'";
  }
}

// LZ (processor-dependent, flang prints leading zero) with F editing
TEST(LeadingZeroTests, LZ_F_editing) {
  static constexpr std::pair<const char *, const char *> cases[]{
      {"(LZ,F6.1)", "   0.2"},
      {"(LZ,F10.3)", "     0.200"},
  };
  double values[]{0.2, 0.2};
  for (int i = 0; i < 2; ++i) {
    std::string got;
    ASSERT_TRUE(
        CompareFormatReal(cases[i].first, values[i], cases[i].second, got))
        << "Failed: format=" << cases[i].first << " value=" << values[i]
        << ", expected '" << cases[i].second << "', got '" << got << "'";
  }
}

// LZP with E editing: value < 1 should print "0." before decimal digits
TEST(LeadingZeroTests, LZP_E_editing) {
  static constexpr std::pair<const char *, const char *> cases[]{
      {"(LZP,E10.3)", " 0.200E+00"},
      {"(LZP,E12.5)", " 0.20000E+00"},
  };
  double values[]{0.2, 0.2};
  for (int i = 0; i < 2; ++i) {
    std::string got;
    ASSERT_TRUE(
        CompareFormatReal(cases[i].first, values[i], cases[i].second, got))
        << "Failed: format=" << cases[i].first << " value=" << values[i]
        << ", expected '" << cases[i].second << "', got '" << got << "'";
  }
}

// LZS with E editing: value < 1 should suppress the leading zero
TEST(LeadingZeroTests, LZS_E_editing) {
  static constexpr std::pair<const char *, const char *> cases[]{
      {"(LZS,E10.3)", "  .200E+00"},
      {"(LZS,E12.5)", "  .20000E+00"},
  };
  double values[]{0.2, 0.2};
  for (int i = 0; i < 2; ++i) {
    std::string got;
    ASSERT_TRUE(
        CompareFormatReal(cases[i].first, values[i], cases[i].second, got))
        << "Failed: format=" << cases[i].first << " value=" << values[i]
        << ", expected '" << cases[i].second << "', got '" << got << "'";
  }
}

// LZP with D editing
TEST(LeadingZeroTests, LZP_D_editing) {
  std::string got;
  ASSERT_TRUE(CompareFormatReal("(LZP,D10.3)", 0.2, " 0.200D+00", got))
      << "Expected ' 0.200D+00', got '" << got << "'";
}

// LZS with D editing
TEST(LeadingZeroTests, LZS_D_editing) {
  std::string got;
  ASSERT_TRUE(CompareFormatReal("(LZS,D10.3)", 0.2, "  .200D+00", got))
      << "Expected '  .200D+00', got '" << got << "'";
}

// LZP with G editing — G routes to F when exponent is in range
TEST(LeadingZeroTests, LZP_G_editing_F_path) {
  std::string got;
  // 0.2 with G10.3: exponent 0 is in [0,3], so G uses F editing
  ASSERT_TRUE(CompareFormatReal("(LZP,G10.3)", 0.2, " 0.200    ", got))
      << "Expected ' 0.200    ', got '" << got << "'";
}

// LZS with G editing — G routes to F when exponent is in range
TEST(LeadingZeroTests, LZS_G_editing_F_path) {
  std::string got;
  ASSERT_TRUE(CompareFormatReal("(LZS,G10.3)", 0.2, "  .200    ", got))
      << "Expected '  .200    ', got '" << got << "'";
}

// LZP with G editing — G routes to E when exponent is out of range
TEST(LeadingZeroTests, LZP_G_editing_E_path) {
  std::string got;
  // 0.0002 with G10.3: exponent -3 is < 0, so G uses E editing
  ASSERT_TRUE(CompareFormatReal("(LZP,G10.3)", 0.0002, " 0.200E-03", got))
      << "Expected ' 0.200E-03', got '" << got << "'";
}

// LZS with G editing — G routes to E when exponent is out of range
TEST(LeadingZeroTests, LZS_G_editing_E_path) {
  std::string got;
  ASSERT_TRUE(CompareFormatReal("(LZS,G10.3)", 0.0002, "  .200E-03", got))
      << "Expected '  .200E-03', got '" << got << "'";
}

// Switching between LZP and LZS in the same format
TEST(LeadingZeroTests, SwitchBetweenLZPandLZS) {
  char buffer[800];
  const char *format{"(LZP,F6.1,LZS,F6.1)"};
  auto cookie{IONAME(BeginInternalFormattedOutput)(
      buffer, sizeof buffer, format, std::strlen(format))};
  EXPECT_TRUE(IONAME(OutputReal64)(cookie, 0.5));
  EXPECT_TRUE(IONAME(OutputReal64)(cookie, 0.5));
  auto status{IONAME(EndIoStatement)(cookie)};
  EXPECT_EQ(status, 0);
  std::string got{buffer, sizeof buffer};
  auto lastNonBlank{got.find_last_not_of(" ")};
  if (lastNonBlank != std::string::npos) {
    got.resize(lastNonBlank + 1);
  }
  std::string expect{"   0.5    .5"};
  ASSERT_TRUE(CompareFormattedStrings(expect, got))
      << "Expected '" << expect << "', got '" << got << "'";
}

// LZP/LZS with negative values < 1 in magnitude
TEST(LeadingZeroTests, NegativeValues) {
  std::string got;
  ASSERT_TRUE(CompareFormatReal("(LZP,F7.1)", -0.2, "   -0.2", got))
      << "Expected '   -0.2', got '" << got << "'";
  ASSERT_TRUE(CompareFormatReal("(LZS,F7.1)", -0.2, "    -.2", got))
      << "Expected '    -.2', got '" << got << "'";
}

// LZP/LZS should not affect values >= 1 (leading zero is not optional)
TEST(LeadingZeroTests, ValuesGreaterThanOne) {
  std::string got;
  ASSERT_TRUE(CompareFormatReal("(LZP,F6.1)", 1.2, "   1.2", got))
      << "Expected '   1.2', got '" << got << "'";
  ASSERT_TRUE(CompareFormatReal("(LZS,F6.1)", 1.2, "   1.2", got))
      << "Expected '   1.2', got '" << got << "'";
  ASSERT_TRUE(CompareFormatReal("(LZP,F6.1)", 12.3, "  12.3", got))
      << "Expected '  12.3', got '" << got << "'";
  ASSERT_TRUE(CompareFormatReal("(LZS,F6.1)", 12.3, "  12.3", got))
      << "Expected '  12.3', got '" << got << "'";
}

// LZP/LZS with zero value
TEST(LeadingZeroTests, ZeroValue) {
  std::string got;
  // LZP: zero value still prints leading zero before decimal point
  ASSERT_TRUE(CompareFormatReal("(LZP,F6.1)", 0.0, "   0.0", got))
      << "Expected '   0.0', got '" << got << "'";
  // LZS: zero has magnitude < 1, so the leading zero is optional and suppressed
  ASSERT_TRUE(CompareFormatReal("(LZS,F6.1)", 0.0, "    .0", got))
      << "Expected '    .0', got '" << got << "'";
}

// LZP/LZS with scale factor (1P) — leading zero not optional when scale > 0
TEST(LeadingZeroTests, WithScaleFactor) {
  std::string got;
  // With 1P, E editing puts one digit before the decimal point,
  // so LZS should not suppress it (it's not an optional zero)
  ASSERT_TRUE(CompareFormatReal("(LZP,1P,E10.3)", 0.2, " 2.000E-01", got))
      << "Expected ' 2.000E-01', got '" << got << "'";
  ASSERT_TRUE(CompareFormatReal("(LZS,1P,E10.3)", 0.2, " 2.000E-01", got))
      << "Expected ' 2.000E-01', got '" << got << "'";
}

// LZP without comma separator (C1302 extension)
TEST(LeadingZeroTests, WithoutCommaSeparator) {
  std::string got;
  ASSERT_TRUE(CompareFormatReal("(LZPF6.1)", 0.2, "   0.2", got))
      << "Expected '   0.2', got '" << got << "'";
  ASSERT_TRUE(CompareFormatReal("(LZSF6.1)", 0.2, "    .2", got))
      << "Expected '    .2', got '" << got << "'";
  ASSERT_TRUE(CompareFormatReal("(LZF6.1)", 0.2, "   0.2", got))
      << "Expected '   0.2', got '" << got << "'";
}
