//===-- runtime/stop.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "stop.h"
#include "file.h"
#include "io-error.h"
#include "terminator.h"
#include "unit.h"
#include <cfenv>
#include <cstdio>
#include <cstdlib>

extern "C" {

static void DescribeIEEESignaledExceptions() {
#ifdef fetestexcept // a macro in some environments; omit std::
  auto excepts{fetestexcept(FE_ALL_EXCEPT)};
#else
  auto excepts{std::fetestexcept(FE_ALL_EXCEPT)};
#endif
  if (excepts) {
    std::fputs("IEEE arithmetic exceptions signaled:", stderr);
    if (excepts & FE_DIVBYZERO) {
      std::fputs(" DIVBYZERO", stderr);
    }
    if (excepts & FE_INEXACT) {
      std::fputs(" INEXACT", stderr);
    }
    if (excepts & FE_INVALID) {
      std::fputs(" INVALID", stderr);
    }
    if (excepts & FE_OVERFLOW) {
      std::fputs(" OVERFLOW", stderr);
    }
    if (excepts & FE_UNDERFLOW) {
      std::fputs(" UNDERFLOW", stderr);
    }
    std::fputc('\n', stderr);
  }
}

static void CloseAllExternalUnits(const char *why) {
  Fortran::runtime::io::IoErrorHandler handler{why};
  Fortran::runtime::io::ExternalFileUnit::CloseAll(handler);
}

[[noreturn]] void RTNAME(StopStatement)(
    int code, bool isErrorStop, bool quiet) {
  CloseAllExternalUnits("STOP statement");
  if (!quiet) {
    std::fprintf(stderr, "Fortran %s", isErrorStop ? "ERROR STOP" : "STOP");
    if (code != EXIT_SUCCESS) {
      std::fprintf(stderr, ": code %d\n", code);
    }
    std::fputc('\n', stderr);
    DescribeIEEESignaledExceptions();
  }
  std::exit(code);
}

[[noreturn]] void RTNAME(StopStatementText)(
    const char *code, bool isErrorStop, bool quiet) {
  CloseAllExternalUnits("STOP statement");
  if (!quiet) {
    std::fprintf(
        stderr, "Fortran %s: %s\n", isErrorStop ? "ERROR STOP" : "STOP", code);
    DescribeIEEESignaledExceptions();
  }
  std::exit(EXIT_FAILURE);
}

void RTNAME(PauseStatement)() {
  if (Fortran::runtime::io::IsATerminal(0)) {
    Fortran::runtime::io::IoErrorHandler handler{"PAUSE statement"};
    Fortran::runtime::io::ExternalFileUnit::FlushAll(handler);
    std::fputs("Fortran PAUSE: hit RETURN to continue:", stderr);
    std::fflush(nullptr);
    if (std::fgetc(stdin) == EOF) {
      CloseAllExternalUnits("PAUSE statement");
      std::exit(EXIT_SUCCESS);
    }
  }
}

[[noreturn]] void RTNAME(FailImageStatement)() {
  Fortran::runtime::NotifyOtherImagesOfFailImageStatement();
  CloseAllExternalUnits("FAIL IMAGE statement");
  std::exit(EXIT_FAILURE);
}

[[noreturn]] void RTNAME(ProgramEndStatement)() {
  CloseAllExternalUnits("END statement");
  std::exit(EXIT_SUCCESS);
}
}
