//===-- unittests/Runtime/Stop.cpp ------------------------------*- C++ -*-===//
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
#include "flang-rt/runtime/environment.h"
#include "flang-rt/runtime/terminator.h"
#include <cstdlib>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>

using namespace Fortran::runtime;

struct TestProgramEnd : CrashHandlerFixture {};

TEST_F(TestProgramEnd, StopTest) {
  EXPECT_EXIT(RTNAME(StopStatement)(), testing::ExitedWithCode(EXIT_SUCCESS),
      "Fortran STOP");
}

TEST_F(TestProgramEnd, StopTestNoStopMessage) {
  EXPECT_EXIT(
      {
        static char noStopMessage[] = "NO_STOP_MESSAGE=1";
        putenv(noStopMessage);
        Fortran::runtime::executionEnvironment.Configure(
            0, nullptr, nullptr, nullptr);
        RTNAME(StopStatement)();
      },
      testing::ExitedWithCode(EXIT_SUCCESS), "");
}

TEST_F(TestProgramEnd, StopMessageTest) {
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

TEST_F(TestProgramEnd, NoStopMessageTest) {
  static const char *message{"bye bye"};
  EXPECT_EXIT(
      {
        static char noStopMessage[] = "NO_STOP_MESSAGE=1";
        putenv(noStopMessage);
        Fortran::runtime::executionEnvironment.Configure(
            0, nullptr, nullptr, nullptr);
        RTNAME(StopStatementText)(message, std::strlen(message),
            /*isErrorStop=*/false, /*quiet=*/false);
      },
      testing::ExitedWithCode(EXIT_SUCCESS), "bye bye");

  EXPECT_EXIT(
      {
        static char noStopMessage[] = "NO_STOP_MESSAGE=1";
        putenv(noStopMessage);
        Fortran::runtime::executionEnvironment.Configure(
            0, nullptr, nullptr, nullptr);
        RTNAME(StopStatementText)(message, std::strlen(message),
            /*isErrorStop=*/false, /*quiet=*/true);
      },
      testing::ExitedWithCode(EXIT_SUCCESS), "");

  EXPECT_EXIT(
      {
        static char noStopMessage[] = "NO_STOP_MESSAGE=1";
        putenv(noStopMessage);
        Fortran::runtime::executionEnvironment.Configure(
            0, nullptr, nullptr, nullptr);
        RTNAME(StopStatementText)(message, std::strlen(message),
            /*isErrorStop=*/true, /*quiet=*/false);
      },
      testing::ExitedWithCode(EXIT_FAILURE), "Fortran ERROR STOP: bye bye");

  EXPECT_EXIT(
      {
        static char noStopMessage[] = "NO_STOP_MESSAGE=1";
        putenv(noStopMessage);
        Fortran::runtime::executionEnvironment.Configure(
            0, nullptr, nullptr, nullptr);
        RTNAME(StopStatementText)(message, std::strlen(message),
            /*isErrorStop=*/true, /*quiet=*/true);
      },
      testing::ExitedWithCode(EXIT_FAILURE), "");
}

TEST_F(TestProgramEnd, FailImageTest) {
  EXPECT_EXIT(
      RTNAME(FailImageStatement)(), testing::ExitedWithCode(EXIT_FAILURE), "");
}

TEST_F(TestProgramEnd, ExitTest) {
  EXPECT_EXIT(RTNAME(Exit)(), testing::ExitedWithCode(EXIT_SUCCESS), "");
  EXPECT_EXIT(
      RTNAME(Exit)(EXIT_FAILURE), testing::ExitedWithCode(EXIT_FAILURE), "");
}

TEST_F(TestProgramEnd, AbortTest) { EXPECT_DEATH(RTNAME(Abort)(), ""); }

TEST_F(TestProgramEnd, CrashTest) {
  static const std::string crashMessage{"bad user code"};
  static const std::string fileName{"file name"};
  // CrashHandlerFixture installs a custom crash handler that formats messages
  // as: "Test <name> crashed in file <file>(<line>): <message>"
  static const std::string fullMessage{"Test CrashTest crashed in file " +
      fileName + "\\(343\\): " + crashMessage};
  EXPECT_DEATH(
      RTNAME(ReportFatalUserError)(crashMessage.c_str(), fileName.c_str(), 343),
      fullMessage.c_str());
}

TEST_F(TestProgramEnd, CrashValuesTest) {
  EXPECT_DEATH(
      RTNAME(ReportFatalUserErrorValues)("sizes %lld, %lld, %lld, %lld %%",
          "check.f90", 42, -17, 4294967296LL, 0, 9),
      "check.f90\\(42\\): sizes -17, 4294967296, 0, 9 %");
}

TEST_F(TestProgramEnd, CrashValuesWithoutValuesTest) {
  EXPECT_DEATH(RTNAME(ReportFatalUserErrorValues)(
                   "bad argument: 100%%", "check.f90", 7, 0, 0, 0, 0),
      "check.f90\\(7\\): bad argument: 100%");
}

TEST_F(TestProgramEnd, CrashValueArrayTest) {
  const long long values[]{std::numeric_limits<long long>::min(),
      std::numeric_limits<long long>::max(), 0, -1, 4294967296LL, 17};
  EXPECT_DEATH(
      RTNAME(ReportFatalUserErrorValueArray)(
          "%lld %lld %lld %lld %lld %%%lld %%lld", "check.f90", 43, 6, values),
      "fatal Fortran runtime error\\(check.f90:43\\): "
      "-9223372036854775808 9223372036854775807 0 -1 "
      "4294967296 %17 %lld");
}

TEST_F(TestProgramEnd, CrashValueArrayWithoutValuesTest) {
  EXPECT_DEATH(RTNAME(ReportFatalUserErrorValueArray)(
                   "bad argument: 100%%", "check.f90", 7, 0, nullptr),
      "fatal Fortran runtime error\\(check.f90:7\\): bad argument: 100%");
  EXPECT_DEATH(
      RTNAME(ReportFatalUserErrorValueArray)("", "check.f90", 7, 0, nullptr),
      "fatal Fortran runtime error\\(check.f90:7\\): ");
}

TEST_F(TestProgramEnd, CrashValueArrayInvalidFormatTest) {
  for (const char *message : {"%", "%l", "%ll", "%d", "%n", "%1$lld"}) {
    EXPECT_DEATH(RTNAME(ReportFatalUserErrorValueArray)(
                     message, "check.f90", 7, 0, nullptr),
        "Invalid fatal error message format");
  }
}

TEST_F(TestProgramEnd, CrashValueArrayCountMismatchTest) {
  const long long values[]{1, 2};
  EXPECT_DEATH(RTNAME(ReportFatalUserErrorValueArray)(
                   "%lld %lld", "check.f90", 7, 1, values),
      "Fatal error message value count mismatch");
  EXPECT_DEATH(
      RTNAME(ReportFatalUserErrorValueArray)("%lld", "check.f90", 7, 2, values),
      "Fatal error message value count mismatch");
  EXPECT_DEATH(RTNAME(ReportFatalUserErrorValueArray)(
                   "%lld", "check.f90", 7, 0, nullptr),
      "Fatal error message value count mismatch");
  EXPECT_DEATH(RTNAME(ReportFatalUserErrorValueArray)(
                   "%lld", "check.f90", 7, 1, nullptr),
      "RUNTIME_CHECK");
}

TEST_F(TestProgramEnd, CrashValueArrayLargeCountMismatchTest) {
  const long long value{0};
  for (std::size_t count : {std::numeric_limits<std::size_t>::max(),
           std::numeric_limits<std::size_t>::max() / 64}) {
    EXPECT_DEATH(RTNAME(ReportFatalUserErrorValueArray)(
                     "%lld", "check.f90", 7, count, &value),
        "Fatal error message value count mismatch");
  }
}

TEST_F(TestProgramEnd, CrashValueArrayStandardDiagnosticTest) {
  // A long literal must span several chunks without truncation.
  std::string message(4096, 'x');
  message += " %lld %lld %lld %lld %lld %%";
  const long long values[]{-1, 0, 1, 2, 3};
  EXPECT_DEATH(RTNAME(ReportFatalUserErrorValueArray)(
                   message.c_str(), "check.f90", 99, 5, values),
      ("fatal Fortran runtime error\\(check.f90:99\\): " +
          std::string(4096, 'x') + " -1 0 1 2 3 %")
          .c_str());
}
