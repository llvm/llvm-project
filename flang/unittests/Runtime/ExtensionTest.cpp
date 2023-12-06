//===-- flang/unittests/Runtime/ExtensionTest.cpp
//---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CrashHandlerFixture.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "flang/Runtime/descriptor.h"
#include "flang/Runtime/extensions.h"
#include "flang/Runtime/main.h"
#include <cstdlib>

#ifdef _WIN32
#include <lmcons.h> // UNLEN
#define LOGIN_NAME_MAX UNLEN
#else
#include <limits.h> // LOGIN_NAME_MAX
#endif

using namespace Fortran::runtime;

struct ExtensionTests : CrashHandlerFixture {};

TEST_F(ExtensionTests, GetlogGetName) {
  const int charLen{3};
  char input[charLen]{"\0\0"};

  FORTRAN_PROCEDURE_NAME(getlog)
  (reinterpret_cast<std::int8_t *>(input), charLen);

  EXPECT_NE(input[0], '\0');
}

TEST_F(ExtensionTests, GetlogPadSpace) {
  // guarantee 1 char longer than max, last char should be pad with space
  const int charLen{LOGIN_NAME_MAX + 2};

  char input[charLen];

  FORTRAN_PROCEDURE_NAME(getlog)
  (reinterpret_cast<std::int8_t *>(input), charLen);

  EXPECT_EQ(input[charLen - 1], ' ');
}