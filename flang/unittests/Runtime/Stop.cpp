//===-- flang/unittests/Runtime/Stop.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// Test runtime API for STOP statement and runtime API to kill the program.
//
//===----------------------------------------------------------------------===//
#include "flang/Runtime/stop.h"
#include "CrashHandlerFixture.h"
#include "../../runtime/environment.h"
#include <cstdlib>
#include <gtest/gtest.h>

using namespace Fortran::runtime;

struct TestProgramEnd : CrashHandlerFixture {};

TEST(TestProgramEnd, StopTest) {
  EXPECT_EXIT(RTNAME(StopStatement)(), testing::ExitedWithCode(EXIT_SUCCESS),
      "Fortran STOP");
}

TEST(TestProgramEnd, StopTestNoStopMessage) {
  putenv(const_cast<char *>("NO_STOP_MESSAGE=1"));
  Fortran::runtime::executionEnvironment.Configure(0, nullptr, nullptr);
  EXPECT_EXIT(
      RTNAME(StopStatement)(), testing::ExitedWithCode(EXIT_SUCCESS), "");
}

TEST(TestProgramEnd, StopMessageTest) {
  static const char *message{"bye bye"};
  EXPECT_EXIT(RTNAME(StopStatementText)(message, std::strlen(message),
                  /*isErrorStop=*/false, /*quiet=*/false),
      testing::ExitedWithCode(EXIT_SUCCESS), "Fortran STOP: bye bye");

  EXPECT_EXIT(RTNAME(StopStatementText)(message, std::strlen(message),
                  /*isErrorStop=*/false, /*quiet=*/true),
      testing::ExitedWithCode(EXIT_SUCCESS), "");

  EXPECT_EXIT(RTNAME(StopStatementText)(message, std::strlen(message),
                  /*isErrorStop=*/true, /*quiet=*/false),
      testing::ExitedWithCode(EXIT_FAILURE), "Fortran ERROR STOP: bye bye");

  EXPECT_EXIT(RTNAME(StopStatementText)(message, std::strlen(message),
                  /*isErrorStop=*/true, /*quiet=*/true),
      testing::ExitedWithCode(EXIT_FAILURE), "");
}

TEST(TestProgramEnd, NoStopMessageTest) {
  putenv(const_cast<char *>("NO_STOP_MESSAGE=1"));
  Fortran::runtime::executionEnvironment.Configure(0, nullptr, nullptr);
  static const char *message{"bye bye"};
  EXPECT_EXIT(RTNAME(StopStatementText)(message, std::strlen(message),
                  /*isErrorStop=*/false, /*quiet=*/false),
      testing::ExitedWithCode(EXIT_SUCCESS), "bye bye");

  EXPECT_EXIT(RTNAME(StopStatementText)(message, std::strlen(message),
                  /*isErrorStop=*/false, /*quiet=*/true),
      testing::ExitedWithCode(EXIT_SUCCESS), "");

  EXPECT_EXIT(RTNAME(StopStatementText)(message, std::strlen(message),
                  /*isErrorStop=*/true, /*quiet=*/false),
      testing::ExitedWithCode(EXIT_FAILURE), "Fortran ERROR STOP: bye bye");

  EXPECT_EXIT(RTNAME(StopStatementText)(message, std::strlen(message),
                  /*isErrorStop=*/true, /*quiet=*/true),
      testing::ExitedWithCode(EXIT_FAILURE), "");
}

TEST(TestProgramEnd, FailImageTest) {
  EXPECT_EXIT(
      RTNAME(FailImageStatement)(), testing::ExitedWithCode(EXIT_FAILURE), "");
}

TEST(TestProgramEnd, ExitTest) {
  EXPECT_EXIT(RTNAME(Exit)(), testing::ExitedWithCode(EXIT_SUCCESS), "");
  EXPECT_EXIT(
      RTNAME(Exit)(EXIT_FAILURE), testing::ExitedWithCode(EXIT_FAILURE), "");
}

TEST(TestProgramEnd, AbortTest) { EXPECT_DEATH(RTNAME(Abort)(), ""); }
