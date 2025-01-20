//===-- runtime/stop.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/stop.h"
#include "config.h"
#include "environment.h"
#include "file.h"
#include "io-error.h"
#include "terminator.h"
#include "unit.h"
#include <cfenv>
#include <cstdio>
#include <cstdlib>

#ifdef HAVE_BACKTRACE
#include BACKTRACE_HEADER
#endif

extern "C" {

static void DescribeIEEESignaledExceptions() {
#ifdef fetestexcept // a macro in some environments; omit std::
  auto excepts{fetestexcept(FE_ALL_EXCEPT)};
#else
  auto excepts{std::fetestexcept(FE_ALL_EXCEPT)};
#endif
  if (excepts) {
    std::fputs("IEEE arithmetic exceptions signaled:", stderr);
#ifdef FE_DIVBYZERO
    if (excepts & FE_DIVBYZERO) {
      std::fputs(" DIVBYZERO", stderr);
    }
#endif
#ifdef FE_INEXACT
    if (excepts & FE_INEXACT) {
      std::fputs(" INEXACT", stderr);
    }
#endif
#ifdef FE_INVALID
    if (excepts & FE_INVALID) {
      std::fputs(" INVALID", stderr);
    }
#endif
#ifdef FE_OVERFLOW
    if (excepts & FE_OVERFLOW) {
      std::fputs(" OVERFLOW", stderr);
    }
#endif
#ifdef FE_UNDERFLOW
    if (excepts & FE_UNDERFLOW) {
      std::fputs(" UNDERFLOW", stderr);
    }
#endif
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
  if (Fortran::runtime::executionEnvironment.noStopMessage && code == 0) {
    quiet = true;
  }
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
    const char *code, std::size_t length, bool isErrorStop, bool quiet) {
  CloseAllExternalUnits("STOP statement");
  if (!quiet) {
    if (Fortran::runtime::executionEnvironment.noStopMessage && !isErrorStop) {
      std::fprintf(stderr, "%.*s\n", static_cast<int>(length), code);
    } else {
      std::fprintf(stderr, "Fortran %s: %.*s\n",
          isErrorStop ? "ERROR STOP" : "STOP", static_cast<int>(length), code);
    }
    DescribeIEEESignaledExceptions();
  }
  if (isErrorStop) {
    std::exit(EXIT_FAILURE);
  } else {
    std::exit(EXIT_SUCCESS);
  }
}

static bool StartPause() {
  if (Fortran::runtime::io::IsATerminal(0)) {
    Fortran::runtime::io::IoErrorHandler handler{"PAUSE statement"};
    Fortran::runtime::io::ExternalFileUnit::FlushAll(handler);
    return true;
  }
  return false;
}

static void EndPause() {
  std::fflush(nullptr);
  if (std::fgetc(stdin) == EOF) {
    CloseAllExternalUnits("PAUSE statement");
    std::exit(EXIT_SUCCESS);
  }
}

void RTNAME(PauseStatement)() {
  if (StartPause()) {
    std::fputs("Fortran PAUSE: hit RETURN to continue:", stderr);
    EndPause();
  }
}

void RTNAME(PauseStatementInt)(int code) {
  if (StartPause()) {
    std::fprintf(stderr, "Fortran PAUSE %d: hit RETURN to continue:", code);
    EndPause();
  }
}

void RTNAME(PauseStatementText)(const char *code, std::size_t length) {
  if (StartPause()) {
    std::fprintf(stderr,
        "Fortran PAUSE %.*s: hit RETURN to continue:", static_cast<int>(length),
        code);
    EndPause();
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

[[noreturn]] void RTNAME(Exit)(int status) {
  CloseAllExternalUnits("CALL EXIT()");
  std::exit(status);
}

static void PrintBacktrace() {
#ifdef HAVE_BACKTRACE
  // TODO: Need to parse DWARF information to print function line numbers
  constexpr int MAX_CALL_STACK{999};
  void *buffer[MAX_CALL_STACK];
  int nptrs{(int)backtrace(buffer, MAX_CALL_STACK)};

  if (char **symbols{backtrace_symbols(buffer, nptrs)}) {
    for (int i = 0; i < nptrs; i++) {
      Fortran::runtime::Terminator{}.PrintCrashArgs("#%d %s\n", i, symbols[i]);
    }
    free(symbols);
  }

#else

  // TODO: Need to implement the version for other platforms.
  Fortran::runtime::Terminator{}.PrintCrashArgs("backtrace is not supported.");

#endif
}

[[noreturn]] void RTNAME(Abort)() {
#ifdef HAVE_BACKTRACE
  PrintBacktrace();
#endif
  std::abort();
}

void FORTRAN_PROCEDURE_NAME(backtrace)() { PrintBacktrace(); }

[[noreturn]] void RTNAME(ReportFatalUserError)(
    const char *message, const char *source, int line) {
  Fortran::runtime::Terminator{source, line}.Crash(message);
}
}
