//===-- unittests/Runtime/InputExtensions.cpp -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CrashHandlerFixture.h"
#include "flang-rt/runtime/descriptor.h"
#include "flang/Runtime/io-api.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <gtest/gtest.h>
#include <tuple>

using namespace Fortran::runtime;
using namespace Fortran::runtime::io;

struct InputExtensionTests : CrashHandlerFixture {};

TEST(InputExtensionTests, SeparatorInField_F) {
  static const struct {
    int get;
    const char *format, *data;
    double expect[3];
  } test[] = {
      {2, "(2F6)", "1.25,3.75,", {1.25, 3.75}},
      {2, "(2F6)", "1.25 ,3.75 ,", {1.25, 3.75}},
      {2, "(DC,2F6)", "1,25;3,75;", {1.25, 3.75}},
      {2, "(DC,2F6)", "1,25 ;3,75 ;", {1.25, 3.75}},
  };
  for (std::size_t j{0}; j < sizeof test / sizeof *test; ++j) {
    auto cookie{IONAME(BeginInternalFormattedInput)(test[j].data,
        std::strlen(test[j].data), test[j].format,
        std::strlen(test[j].format))};
    for (int k{0}; k < test[j].get; ++k) {
      float got;
      IONAME(InputReal32)(cookie, got);
      ASSERT_EQ(got, test[j].expect[k])
          << "expected " << test[j].expect[k] << ", got " << got;
    }
    auto status{IONAME(EndIoStatement)(cookie)};
    ASSERT_EQ(status, 0) << "error status " << status << " on F test case "
                         << j;
  }
}

TEST(InputExtensionTests, SeparatorInField_I) {
  static const struct {
    int get;
    const char *format, *data;
    std::int64_t expect[3];
  } test[] = {
      {2, "(2I4)", "12,34,", {12, 34}},
      {2, "(2I4)", "12 ,34 ,", {12, 34}},
      {2, "(DC,2I4)", "12;34;", {12, 34}},
      {2, "(DC,2I4)", "12 ;34 ;", {12, 34}},
  };
  for (std::size_t j{0}; j < sizeof test / sizeof *test; ++j) {
    auto cookie{IONAME(BeginInternalFormattedInput)(test[j].data,
        std::strlen(test[j].data), test[j].format,
        std::strlen(test[j].format))};
    for (int k{0}; k < test[j].get; ++k) {
      std::int64_t got;
      IONAME(InputInteger)(cookie, got);
      ASSERT_EQ(got, test[j].expect[k])
          << "expected " << test[j].expect[k] << ", got " << got;
    }
    auto status{IONAME(EndIoStatement)(cookie)};
    ASSERT_EQ(status, 0) << "error status " << status << " on I test case "
                         << j;
  }
}

TEST(InputExtensionTests, SeparatorInField_L) {
  static const struct {
    int get;
    const char *format, *data;
    bool expect[3];
  } test[] = {
      {2, "(2L4)", ".T,F,", {true, false}},
      {2, "(2L4)", ".F,T,", {false, true}},
      {2, "(2L4)", ".T.,F,", {true, false}},
      {2, "(2L4)", ".F.,T,", {false, true}},
      {2, "(DC,2L4)", ".T;F,", {true, false}},
      {2, "(DC,2L4)", ".F;T,", {false, true}},
      {2, "(DC,2L4)", ".T.;F,", {true, false}},
      {2, "(DC,2L4)", ".F.;T,", {false, true}},
      {2, "(2B4)", "1   0,", {true, false}},
      {2, "(2B4)", "0   1,", {false, true}},
      {2, "(DC,2B4)", "1   0,", {true, false}},
      {2, "(DC,2B4)", "0   1,", {false, true}},
  };
  for (std::size_t j{0}; j < sizeof test / sizeof *test; ++j) {
    auto cookie{IONAME(BeginInternalFormattedInput)(test[j].data,
        std::strlen(test[j].data), test[j].format,
        std::strlen(test[j].format))};
    for (int k{0}; k < test[j].get; ++k) {
      bool got;
      IONAME(InputLogical)(cookie, got);
      ASSERT_EQ(got, test[j].expect[k])
          << "expected " << test[j].expect[k] << ", got " << got;
    }
    auto status{IONAME(EndIoStatement)(cookie)};
    ASSERT_EQ(status, 0) << "error status " << status << " on L/B test case "
                         << j;
  }
}

// A tab in a numeric or logical input field is treated as a blank (see
// flang/docs/Extensions.md): ignored under BN, read as a zero under BZ.
TEST(InputExtensionTests, TabInField_I) {
  static const struct {
    const char *format, *data;
    std::int64_t expect;
  } test[] = {
      {"(I4)", "\t123", 123},
      {"(I4)", "1\t23", 123},
      {"(BN,I4)", "\t123", 123},
      {"(BN,I4)", "1\t23", 123},
      {"(BN,I4)", "123\t", 123},
      {"(BN,I4)", "-\t12", -12},
      {"(BN,I4)", "\t\t\t\t", 0},
      {"(BZ,I4)", "\t123", 123},
      {"(BZ,I4)", "1\t23", 1023},
      {"(BZ,I4)", "123\t", 1230},
      {"(BZ,I4)", "\t\t\t\t", 0},
  };
  for (std::size_t j{0}; j < sizeof test / sizeof *test; ++j) {
    auto cookie{IONAME(BeginInternalFormattedInput)(test[j].data,
        std::strlen(test[j].data), test[j].format,
        std::strlen(test[j].format))};
    std::int64_t got{-1};
    IONAME(InputInteger)(cookie, got);
    ASSERT_EQ(got, test[j].expect)
        << "expected " << test[j].expect << ", got " << got;
    auto status{IONAME(EndIoStatement)(cookie)};
    ASSERT_EQ(status, 0) << "error status " << status << " on I test case "
                         << j;
  }
}

TEST(InputExtensionTests, TabInField_F) {
  static const struct {
    const char *format, *data;
    double expect;
  } test[] = {
      {"(F4.0)", "\t123", 123},
      {"(F4.0)", "1\t23", 123},
      {"(BN,F4.0)", "\t123", 123},
      {"(BN,F4.0)", "1\t23", 123},
      {"(BN,F4.0)", "123\t", 123},
      {"(BN,F4.0)", "-\t12", -12},
      {"(BN,F4.0)", "\t\t\t\t", 0},
      {"(BZ,F4.0)", "\t123", 123},
      {"(BZ,F4.0)", "1\t23", 1023},
      {"(BZ,F4.0)", "123\t", 1230},
      {"(BZ,F4.0)", "\t\t\t\t", 0},
  };
  for (std::size_t j{0}; j < sizeof test / sizeof *test; ++j) {
    auto cookie{IONAME(BeginInternalFormattedInput)(test[j].data,
        std::strlen(test[j].data), test[j].format,
        std::strlen(test[j].format))};
    float got{-1};
    IONAME(InputReal32)(cookie, got);
    ASSERT_EQ(got, test[j].expect)
        << "expected " << test[j].expect << ", got " << got;
    auto status{IONAME(EndIoStatement)(cookie)};
    ASSERT_EQ(status, 0) << "error status " << status << " on F test case "
                         << j;
  }
}

// Tabs around the digits, decimal point and exponent of real literals.
TEST(InputExtensionTests, TabInField_F_Literals) {
  static const struct {
    const char *format, *data;
    double expect;
  } test[] = {
      {"(BN,F5.0)", "\t3.14", 3.14},
      {"(BN,F5.0)", "3.\t14", 3.14},
      {"(BN,F5.0)", "3.14\t", 3.14},
      {"(BZ,F5.0)", "3.\t14", 3.014},
      {"(BN,F7.0)", "\t1.2E-3", 1.2E-3},
      {"(BN,F7.0)", "1.2E-\t3", 1.2E-3},
      {"(BN,F7.0)", "1.2E-3\t", 1.2E-3},
      {"(BZ,F7.0)", "1.2E-3\t", 1.2E-30},
      {"(BN,F3.0)", "\t.7", 0.7},
      {"(BN,F3.0)", ".\t7", 0.7},
      {"(BZ,F3.0)", ".\t7", 0.07},
  };
  for (std::size_t j{0}; j < sizeof test / sizeof *test; ++j) {
    auto cookie{IONAME(BeginInternalFormattedInput)(test[j].data,
        std::strlen(test[j].data), test[j].format,
        std::strlen(test[j].format))};
    double got{-1};
    IONAME(InputReal64)(cookie, got);
    ASSERT_EQ(got, test[j].expect)
        << "expected " << test[j].expect << ", got " << got;
    auto status{IONAME(EndIoStatement)(cookie)};
    ASSERT_EQ(status, 0) << "error status " << status << " on F test case "
                         << j;
  }
}

TEST(InputExtensionTests, TabInField_L) {
  static const struct {
    const char *format, *data;
    bool expect;
  } test[] = {
      {"(L2)", "\tT", true},
      {"(L2)", "\tF", false},
      {"(L2)", "T\t", true},
      {"(L3)", "\t.F", false},
  };
  for (std::size_t j{0}; j < sizeof test / sizeof *test; ++j) {
    auto cookie{IONAME(BeginInternalFormattedInput)(test[j].data,
        std::strlen(test[j].data), test[j].format,
        std::strlen(test[j].format))};
    bool got{!test[j].expect};
    IONAME(InputLogical)(cookie, got);
    ASSERT_EQ(got, test[j].expect)
        << "expected " << test[j].expect << ", got " << got;
    auto status{IONAME(EndIoStatement)(cookie)};
    ASSERT_EQ(status, 0) << "error status " << status << " on L test case "
                         << j;
  }
}

// Character (A) input transfers a tab unchanged.
TEST(InputExtensionTests, TabInField_A) {
  static const char data[]{"X\tY"};
  static const char format[]{"(A3)"};
  auto cookie{IONAME(BeginInternalFormattedInput)(
      data, std::strlen(data), format, std::strlen(format))};
  char got[3]{'?', '?', '?'};
  ASSERT_TRUE(IONAME(InputAscii)(cookie, got, sizeof got));
  for (std::size_t k{0}; k < sizeof got; ++k) {
    ASSERT_EQ(got[k], data[k]) << "at position " << k;
  }
  auto status{IONAME(EndIoStatement)(cookie)};
  ASSERT_EQ(status, 0) << "error status " << status;
}

// A tab also separates values in list-directed input.
TEST(InputExtensionTests, TabSeparator_ListDirected) {
  static const char data[]{"1\t2 \t3"};
  auto cookie{IONAME(BeginInternalListInput)(data, std::strlen(data))};
  for (std::int64_t expect{1}; expect <= 3; ++expect) {
    std::int64_t got{-1};
    IONAME(InputInteger)(cookie, got);
    ASSERT_EQ(got, expect) << "expected " << expect << ", got " << got;
  }
  auto status{IONAME(EndIoStatement)(cookie)};
  ASSERT_EQ(status, 0) << "error status " << status;
}
