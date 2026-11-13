//===-- lib/runtime/stop.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/stop.h"
#include "config.h"
#include "flang-rt/runtime/environment.h"
#include "flang-rt/runtime/file.h"
#include "flang-rt/runtime/io-error.h"
#include "flang-rt/runtime/terminator.h"
#include "flang-rt/runtime/unit.h"
#include <cfenv>
#include <cstdio>
#include <cstdlib>

#ifdef HAVE_BACKTRACE
#include BACKTRACE_HEADER
#endif

extern "C" {

[[maybe_unused]] static void DescribeIEEESignaledExceptions() {
#if defined(RT_DEVICE_COMPILATION) || RT_GPU_TARGET
  unsigned excepts{}; // No fenv support on the device.
#else
#ifdef fetestexcept // a macro in some environments; omit std::
  auto excepts{fetestexcept(FE_ALL_EXCEPT)};
#else
  auto excepts{std::fetestexcept(FE_ALL_EXCEPT)};
#endif
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
#if !RT_GPU_TARGET
  Fortran::runtime::io::IoErrorHandler handler{why};
  Fortran::runtime::io::ExternalFileUnit::CloseAll(handler);
#endif
}

[[noreturn]] RT_API_ATTRS void RTNAME(StopStatement)(
    int code, bool isErrorStop, bool quiet) {
#if defined(RT_DEVICE_COMPILATION)
  if (Fortran::runtime::executionEnvironment.noStopMessage && code == 0) {
    quiet = true;
  }
  if (!quiet) {
    if (isErrorStop) {
      std::printf("Fortran ERROR STOP");
    } else {
      std::printf("Fortran STOP");
    }
    if (code != EXIT_SUCCESS) {
      std::printf(": code %d\n", code);
    }
    std::printf("\n");
  }
  Fortran::runtime::DeviceTrap();
#else
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
  if (isErrorStop) {
#ifdef FLANG_RT_HAVE_CSIGNAL
    Fortran::runtime::PrintFatalBacktrace("\nError termination. Backtrace:\n");
#endif
    Fortran::runtime::ErrorExit(code);
  } else {
    Fortran::runtime::NormalExit(code);
  }
#endif
}

[[noreturn]] RT_API_ATTRS void RTNAME(StopStatementText)(
    const char *code, std::size_t length, bool isErrorStop, bool quiet) {
#if defined(RT_DEVICE_COMPILATION)
  if (!quiet) {
    if (Fortran::runtime::executionEnvironment.noStopMessage && !isErrorStop) {
      std::printf("%s\n", code);
    } else {
      std::printf(
          "Fortran %s: %s\n", isErrorStop ? "ERROR STOP" : "STOP", code);
    }
  }
  Fortran::runtime::DeviceTrap();
#else
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
#ifdef FLANG_RT_HAVE_CSIGNAL
    Fortran::runtime::PrintFatalBacktrace("\nError termination. Backtrace:\n");
#endif
    Fortran::runtime::ErrorExit(EXIT_FAILURE);
  } else {
    Fortran::runtime::NormalExit(EXIT_SUCCESS);
  }
#endif
}

#if !RT_GPU_TARGET
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
    Fortran::runtime::ErrorExit(EXIT_SUCCESS);
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
#endif

[[noreturn]] void RTNAME(FailImageStatement)() {
  CloseAllExternalUnits("FAIL IMAGE statement");
  Fortran::runtime::NotifyOtherImagesOfFailImageStatement();
  Fortran::runtime::NormalExit(EXIT_FAILURE);
}

[[noreturn]] void RTNAME(ProgramEndStatement)() {
  CloseAllExternalUnits("END statement");
  Fortran::runtime::NormalExit(EXIT_SUCCESS);
}

void RTNAME(RegisterImagesNormalEndCallback)(void (*callback)(int)) {
  Fortran::runtime::SetNormalEndCallback(callback);
}

void RTNAME(RegisterImagesErrorCallback)(void (*callback)(int)) {
  Fortran::runtime::SetErrorCallback(callback);
}

void RTNAME(RegisterFailImageCallback)(void (*callback)(void)) {
  Fortran::runtime::SetFailImageCallback(callback);
}

[[noreturn]] void RTNAME(Exit)(int status) {
  CloseAllExternalUnits("CALL EXIT()");
  Fortran::runtime::NormalExit(status);
}

static RT_NOINLINE_ATTR void PrintBacktrace() {
#ifdef HAVE_BACKTRACE
  // TODO: Need to parse DWARF information to print function line numbers
  constexpr int MAX_CALL_STACK{999};
  void *buffer[MAX_CALL_STACK];
  int nptrs{(int)backtrace(buffer, MAX_CALL_STACK)};

  if (char **symbols{backtrace_symbols(buffer, nptrs)}) {
    // Skip the PrintBacktrace() frame, as it is just a utility.
    // It makes sense to start printing the backtrace
    // from Abort() or backtrace().
    for (int i = 1; i < nptrs; i++) {
      Fortran::runtime::Terminator{}.PrintCrashArgs(
          "#%d %s\n", i - 1, symbols[i]);
    }
    free(symbols);
  }

#else

  // TODO: Need to implement the version for other platforms.
  Fortran::runtime::Terminator{}.PrintCrashArgs("backtrace is not supported.");

#endif
}

[[noreturn]] RT_OPTNONE_ATTR void RTNAME(Abort)() {
#ifdef FLANG_RT_HAVE_CSIGNAL
  Fortran::runtime::PrintFatalBacktrace("\nProgram aborted. Backtrace:\n");
  // The fatal-signal handler also traces SIGABRT. Reset it so abort() does
  // not print a second trace.
  std::signal(SIGABRT, SIG_DFL);
#endif
  std::abort();
}

RT_OPTNONE_ATTR void FORTRAN_PROCEDURE_NAME(backtrace)() { PrintBacktrace(); }

RT_EXT_API_GROUP_BEGIN
[[noreturn]] RT_API_ATTRS void RTNAME(ReportFatalUserError)(
    const char *message, const char *source, int line) {
  Fortran::runtime::Terminator{source, line}.Crash(message);
}

[[noreturn]] RT_API_ATTRS void RTNAME(ReportFatalUserErrorValues)(
    const char *message, const char *source, int line, long long v0,
    long long v1, long long v2, long long v3) {
  Fortran::runtime::Terminator{source, line}.Crash(message, v0, v1, v2, v3);
}

[[noreturn]] RT_API_ATTRS void RTNAME(ReportFatalUserErrorValueArray)(
    const char *message, const char *source, int line, std::size_t count,
    const long long *values) {
  Fortran::runtime::Terminator terminator{source, line};
  RUNTIME_CHECK(terminator, message && (count == 0 || values));

  // Validate the format and count before printing the diagnostic or reading
  // values.
  std::size_t conversions{0};
  for (const char *p{message}; *p; ++p) {
    if (*p == '%') {
      if (p[1] == '%') {
        ++p;
      } else if (p[1] == 'l' && p[2] == 'l' && p[3] == 'd') {
        ++conversions;
        p += 3;
      } else {
        terminator.Crash("Invalid fatal error message format");
      }
    }
  }
  if (conversions != count) {
    terminator.Crash("Fatal error message value count mismatch");
  }

  // Print in pieces so reporting the error does not require allocating a
  // formatted message. Concurrent failures may interleave their output.
  terminator.CrashHeader();
  std::size_t nextValue{0};
  for (const char *p{message}; *p;) {
    if (*p != '%') {
      char chunk[256];
      std::size_t length{0};
      while (*p && *p != '%' && length < sizeof chunk - 1) {
        chunk[length++] = *p++;
      }
      chunk[length] = '\0';
      terminator.PrintCrashArgs("%s", chunk);
    } else if (p[1] == '%') {
      terminator.PrintCrashArgs("%%");
      p += 2;
    } else {
      terminator.PrintCrashArgs("%lld", values[nextValue++]);
      p += 4;
    }
  }
  terminator.CrashFooter();
}
RT_EXT_API_GROUP_END
}

#ifdef FLANG_RT_HAVE_CSIGNAL
void Fortran::runtime::PrintFatalBacktrace(const char *heading) {
  if (!Fortran::runtime::executionEnvironment.backtrace)
    return;
  if (heading)
    std::fputs(heading, stderr);
  PrintBacktrace();
}

static volatile std::sig_atomic_t fatalErrorInProgress{0};

static const char *SignalDescription(int signum) {
  switch (signum) {
#ifdef SIGQUIT
  case SIGQUIT:
    return "SIGQUIT: Terminal quit signal";
#endif
  case SIGILL:
    return "SIGILL: Illegal instruction";
  case SIGABRT:
    return "SIGABRT: Process abort signal";
  case SIGFPE:
    return "SIGFPE: Floating-point exception";
  case SIGSEGV:
    return "SIGSEGV: Segmentation fault";
#ifdef SIGBUS
  case SIGBUS:
    return "SIGBUS: Access to an undefined portion of a memory object";
#endif
#ifdef SIGSYS
  case SIGSYS:
    return "SIGSYS: Bad system call";
#endif
#ifdef SIGTRAP
  case SIGTRAP:
    return "SIGTRAP: Trace/breakpoint trap";
#endif
#ifdef SIGXCPU
  case SIGXCPU:
    return "SIGXCPU: CPU time limit exceeded";
#endif
#ifdef SIGXFSZ
  case SIGXFSZ:
    return "SIGXFSZ: File size limit exceeded";
#endif
  default:
    return nullptr;
  }
}

static void BacktraceSignalHandler(int signum) {
  if (fatalErrorInProgress)
    std::raise(signum);
  fatalErrorInProgress = 1;
  if (const char *desc{SignalDescription(signum)}) {
    std::fprintf(stderr, "\nProgram received signal %s.\n", desc);
  } else {
    std::fprintf(stderr, "\nProgram received signal %d.\n", signum);
  }
  std::fputs("\nBacktrace for this error:\n", stderr);
  PrintBacktrace();
  std::signal(signum, SIG_DFL);
  std::raise(signum);
}

static void InstallOneBacktraceHandler(int signum) {
  std::signal(signum, BacktraceSignalHandler);
}

void Fortran::runtime::InstallFatalBacktraceHandlers() {
#ifdef SIGQUIT
  InstallOneBacktraceHandler(SIGQUIT);
#endif
  InstallOneBacktraceHandler(SIGILL);
  InstallOneBacktraceHandler(SIGABRT);
  InstallOneBacktraceHandler(SIGFPE);
  InstallOneBacktraceHandler(SIGSEGV);
#ifdef SIGBUS
  InstallOneBacktraceHandler(SIGBUS);
#endif
#ifdef SIGSYS
  InstallOneBacktraceHandler(SIGSYS);
#endif
#ifdef SIGTRAP
  InstallOneBacktraceHandler(SIGTRAP);
#endif
#ifdef SIGXCPU
  InstallOneBacktraceHandler(SIGXCPU);
#endif
#ifdef SIGXFSZ
  InstallOneBacktraceHandler(SIGXFSZ);
#endif
}
#endif
