//===-- flang/unittests/Runtime/CrashHandlerFixture.cpp ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "CrashHandlerFixture.h"
#include "flang/../../runtime/terminator.h"
#include <cstdarg>
#include <cstdlib>

// Replaces Fortran runtime's crash handler so we can verify the crash message
[[noreturn]] static void CatchCrash(const char *sourceFile, int sourceLine,
                                    const char *message, va_list &ap) {
  char buffer[1000];
  std::vsnprintf(buffer, sizeof buffer, message, ap);
  va_end(ap);
  llvm::errs()
      << "Test "
      << ::testing::UnitTest::GetInstance()->current_test_info()->name()
      << " crashed in file "
      << (sourceFile ? sourceFile : "unknown source file") << '(' << sourceLine
      << "): " << buffer << '\n';
  std::exit(EXIT_FAILURE);
}

// Register the crash handler above when creating each unit test in this suite
void CrashHandlerFixture::SetUp() {
  static bool isCrashHanlderRegistered{false};

  if (!isCrashHanlderRegistered) {
    Fortran::runtime::Terminator::RegisterCrashHandler(CatchCrash);
  }

  isCrashHanlderRegistered = true;
}
