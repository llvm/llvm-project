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
    ASSERT_EQ(status, 0) << "error status " << status << " on L test case "
                         << j;
  }
}
