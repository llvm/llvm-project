//===-- Implementation of libc death test executors -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LibcTest.h"

#include "test/UnitTest/ExecuteFunction.h"
#include "test/UnitTest/TestLogger.h"

#include <cassert>

namespace __llvm_libc {
namespace testing {

bool Test::testProcessKilled(testutils::FunctionCaller *Func, int Signal,
                             const char *LHSStr, const char *RHSStr,
                             const char *File, unsigned long Line) {
  testutils::ProcessStatus Result = testutils::invoke_in_subprocess(Func, 500);

  if (const char *error = Result.get_error()) {
    Ctx->markFail();
    tlog << File << ":" << Line << ": FAILURE\n" << error << '\n';
    return false;
  }

  if (Result.timed_out()) {
    Ctx->markFail();
    tlog << File << ":" << Line << ": FAILURE\n"
         << "Process timed out after " << 500 << " milliseconds.\n";
    return false;
  }

  if (Result.exited_normally()) {
    Ctx->markFail();
    tlog << File << ":" << Line << ": FAILURE\n"
         << "Expected " << LHSStr
         << " to be killed by a signal\nBut it exited normally!\n";
    return false;
  }

  int KilledBy = Result.get_fatal_signal();
  assert(KilledBy != 0 && "Not killed by any signal");
  if (Signal == -1 || KilledBy == Signal)
    return true;

  using testutils::signal_as_string;
  Ctx->markFail();
  tlog << File << ":" << Line << ": FAILURE\n"
       << "              Expected: " << LHSStr << '\n'
       << "To be killed by signal: " << Signal << '\n'
       << "              Which is: " << signal_as_string(Signal) << '\n'
       << "  But it was killed by: " << KilledBy << '\n'
       << "              Which is: " << signal_as_string(KilledBy) << '\n';
  return false;
}

bool Test::testProcessExits(testutils::FunctionCaller *Func, int ExitCode,
                            const char *LHSStr, const char *RHSStr,
                            const char *File, unsigned long Line) {
  testutils::ProcessStatus Result = testutils::invoke_in_subprocess(Func, 500);

  if (const char *error = Result.get_error()) {
    Ctx->markFail();
    tlog << File << ":" << Line << ": FAILURE\n" << error << '\n';
    return false;
  }

  if (Result.timed_out()) {
    Ctx->markFail();
    tlog << File << ":" << Line << ": FAILURE\n"
         << "Process timed out after " << 500 << " milliseconds.\n";
    return false;
  }

  if (!Result.exited_normally()) {
    Ctx->markFail();
    tlog << File << ":" << Line << ": FAILURE\n"
         << "Expected " << LHSStr << '\n'
         << "to exit with exit code " << ExitCode << '\n'
         << "But it exited abnormally!\n";
    return false;
  }

  int ActualExit = Result.get_exit_code();
  if (ActualExit == ExitCode)
    return true;

  Ctx->markFail();
  tlog << File << ":" << Line << ": FAILURE\n"
       << "Expected exit code of: " << LHSStr << '\n'
       << "             Which is: " << ActualExit << '\n'
       << "       To be equal to: " << RHSStr << '\n'
       << "             Which is: " << ExitCode << '\n';
  return false;
}

} // namespace testing
} // namespace __llvm_libc
