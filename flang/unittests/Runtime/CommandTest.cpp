//===-- flang/unittests/RuntimeGTest/CommandTest.cpp ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/command.h"
#include "gtest/gtest.h"
#include "flang/Runtime/descriptor.h"
#include "flang/Runtime/main.h"

using namespace Fortran::runtime;

class CommandFixture : public ::testing::Test {
protected:
  CommandFixture(int argc, const char *argv[]) {
    RTNAME(ProgramStart)(argc, argv, {});
  }
};

static const char *commandOnlyArgv[]{"aProgram"};
class ZeroArguments : public CommandFixture {
protected:
  ZeroArguments() : CommandFixture(1, commandOnlyArgv) {}
};

TEST_F(ZeroArguments, ArgumentCount) { EXPECT_EQ(0, RTNAME(ArgumentCount)()); }

TEST_F(ZeroArguments, ArgumentLength) {
  EXPECT_EQ(0, RTNAME(ArgumentLength)(-1));
  EXPECT_EQ(8, RTNAME(ArgumentLength)(0));
  EXPECT_EQ(0, RTNAME(ArgumentLength)(1));
}

static const char *oneArgArgv[]{"aProgram", "anArgumentOfLength20"};
class OneArgument : public CommandFixture {
protected:
  OneArgument() : CommandFixture(2, oneArgArgv) {}
};

TEST_F(OneArgument, ArgumentCount) { EXPECT_EQ(1, RTNAME(ArgumentCount)()); }

TEST_F(OneArgument, ArgumentLength) {
  EXPECT_EQ(0, RTNAME(ArgumentLength)(-1));
  EXPECT_EQ(8, RTNAME(ArgumentLength)(0));
  EXPECT_EQ(20, RTNAME(ArgumentLength)(1));
  EXPECT_EQ(0, RTNAME(ArgumentLength)(2));
}

static const char *severalArgsArgv[]{
    "aProgram", "16-char-long-arg", "", "-22-character-long-arg", "o"};
class SeveralArguments : public CommandFixture {
protected:
  SeveralArguments()
      : CommandFixture(sizeof(severalArgsArgv) / sizeof(*severalArgsArgv),
            severalArgsArgv) {}
};

TEST_F(SeveralArguments, ArgumentCount) {
  EXPECT_EQ(4, RTNAME(ArgumentCount)());
}

TEST_F(SeveralArguments, ArgumentLength) {
  EXPECT_EQ(0, RTNAME(ArgumentLength)(-1));
  EXPECT_EQ(8, RTNAME(ArgumentLength)(0));
  EXPECT_EQ(16, RTNAME(ArgumentLength)(1));
  EXPECT_EQ(0, RTNAME(ArgumentLength)(2));
  EXPECT_EQ(22, RTNAME(ArgumentLength)(3));
  EXPECT_EQ(1, RTNAME(ArgumentLength)(4));
  EXPECT_EQ(0, RTNAME(ArgumentLength)(5));
}
