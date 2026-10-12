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
#include "flang-rt/runtime/memory.h"
#include "flang-rt/runtime/terminator.h"
#include "flang-rt/runtime/unit.h"
#include "flang/Runtime/freestanding-tools.h"
#include <cfenv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

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
  if (isErrorStop)
    Fortran::runtime::ErrorExit(code);
  else
    Fortran::runtime::NormalExit(code);
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
#ifdef HAVE_BACKTRACE
  PrintBacktrace();
#endif
  std::abort();
}

RT_OPTNONE_ATTR void FORTRAN_PROCEDURE_NAME(backtrace)() { PrintBacktrace(); }

[[noreturn]] RT_API_ATTRS void RTNAME(ReportFatalUserError)(
    const char *message, const char *source, int line) {
  Fortran::runtime::Terminator{source, line}.Crash(message);
}

RT_OFFLOAD_API_GROUP_BEGIN
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

  // Reserve enough space for every value, its sign, and the unchanged format.
  // Allocation is confined to this fatal path, and no diagnostic is truncated.
  constexpr std::size_t maxValueChars{
      std::numeric_limits<unsigned long long>::digits10 + 2};
  std::size_t messageLength{Fortran::runtime::strlen(message)};
  if (messageLength == std::numeric_limits<std::size_t>::max() ||
      count > (std::numeric_limits<std::size_t>::max() - messageLength - 1) /
              maxValueChars) {
    terminator.Crash("Fatal error message size overflow");
  }
  char *buffer{static_cast<char *>(Fortran::runtime::AllocateMemoryOrCrash(
      terminator, messageLength + count * maxValueChars + 1))};
  char *out{buffer};
  std::size_t nextValue{0};
  for (const char *p{message}; *p; ++p) {
    if (*p != '%') {
      *out++ = *p;
    } else if (p[1] == '%') {
      *out++ = '%';
      ++p;
    } else if (p[1] == 'l' && p[2] == 'l' && p[3] == 'd') {
      if (nextValue == count) {
        terminator.Crash("Fatal error message value count mismatch");
      }
      long long value{values[nextValue++]};
      unsigned long long magnitude{static_cast<unsigned long long>(value)};
      if (value < 0) {
        *out++ = '-';
        // Negate in unsigned arithmetic, including for the most negative value.
        magnitude = -magnitude;
      }
      char digits[maxValueChars];
      char *end{digits + sizeof digits}, *begin{end};
      do {
        *--begin = '0' + magnitude % 10;
        magnitude /= 10;
      } while (magnitude != 0);
      std::size_t length{static_cast<std::size_t>(end - begin)};
      Fortran::runtime::memcpy(out, begin, length);
      out += length;
      p += 3;
    } else {
      terminator.Crash("Invalid fatal error message format");
    }
  }
  if (nextValue != count) {
    terminator.Crash("Fatal error message value count mismatch");
  }
  *out = '\0';
  // Keep both the standard diagnostic and the host crash-handler hook. The
  // rendered percent signs are data, not additional format conversions.
  terminator.Crash("%s", buffer);
}
RT_OFFLOAD_API_GROUP_END
}
