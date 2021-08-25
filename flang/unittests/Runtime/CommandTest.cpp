//===-- flang/unittests/RuntimeGTest/CommandTest.cpp ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../../runtime/command.h"
#include "gtest/gtest.h"
#include "../../runtime/descriptor.h"
#include "../../runtime/main.h"

using namespace Fortran::runtime;

TEST(ArgumentCount, ZeroArguments) {
  const char *argv[]{"aProgram"};
  RTNAME(ProgramStart)(1, argv, {});
  EXPECT_EQ(0, RTNAME(ArgumentCount)());
}

TEST(ArgumentCount, OneArgument) {
  const char *argv[]{"aProgram", "anArgument"};
  RTNAME(ProgramStart)(2, argv, {});
  EXPECT_EQ(1, RTNAME(ArgumentCount)());
}

TEST(ArgumentCount, SeveralArguments) {
  const char *argv[]{"aProgram", "arg1", "arg2", "arg3", "arg4"};
  RTNAME(ProgramStart)(5, argv, {});
  EXPECT_EQ(4, RTNAME(ArgumentCount)());
}
