//===-- flang/unittests/Runtime/LogicalFormatTest.cpp -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CrashHandlerFixture.h"
#include "flang/Runtime/descriptor.h"
#include "flang/Runtime/io-api.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <gtest/gtest.h>
#include <tuple>

using namespace Fortran::runtime;
using namespace Fortran::runtime::io;

TEST(IOApiTests, LogicalFormatTest) {
  static constexpr int bufferSize{29};
  char buffer[bufferSize];

  // Create format for all types and values to be written
  const char *format{"(L,L3,I3,L2,L2,I3,L2,A3,L2,L,F4.1,L2)"};
  auto cookie{IONAME(BeginInternalFormattedOutput)(
      buffer, bufferSize, format, std::strlen(format))};

  // Write string, integer, and logical values to buffer
  IONAME(OutputLogical)(cookie, true);
  IONAME(OutputLogical)(cookie, false);
  IONAME(OutputInteger64)(cookie, 6);
  IONAME(OutputInteger32)(cookie, 22);
  IONAME(OutputInteger32)(cookie, 0);
  IONAME(OutputInteger32)(cookie, -2);
  IONAME(OutputCharacter)(cookie, "ABC", 3);
  IONAME(OutputCharacter)(cookie, "AB", 2);
  IONAME(OutputReal64)(cookie, 0.0);
  IONAME(OutputCharacter)(cookie, "", 0);
  IONAME(OutputReal32)(cookie, 2.3);
  IONAME(OutputReal32)(cookie, 2.3);

  // Ensure IO succeeded
  auto status{IONAME(EndIoStatement)(cookie)};
  ASSERT_EQ(status, 0) << "logical format: '" << format << "' failed, status "
                       << static_cast<int>(status);

  // Ensure final buffer matches expected string output
  static const std::string expect{"T  F  6 T F -2 T AB FF 2.3 T"};

  // expect.size() == bufferSize - 1
  std::string bufferStr = std::string(buffer, bufferSize - 1);
  ASSERT_TRUE(expect == bufferStr)
      << "Expected '" << expect << "', got '" << bufferStr << "'";
}
