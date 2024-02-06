//===-- runtime/terminator.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Termination of the image

#ifndef FORTRAN_RUNTIME_TERMINATOR_H_
#define FORTRAN_RUNTIME_TERMINATOR_H_

#include "flang/Runtime/api-attrs.h"
#include <cstdarg>
#include <cstdio>
#include <cstdlib>

namespace Fortran::runtime {

// A mixin class for statement-specific image error termination
// for errors detected in the runtime library
class Terminator {
public:
  RT_API_ATTRS Terminator() {}
  Terminator(const Terminator &) = default;
  explicit RT_API_ATTRS Terminator(
      const char *sourceFileName, int sourceLine = 0)
      : sourceFileName_{sourceFileName}, sourceLine_{sourceLine} {}

  RT_API_ATTRS const char *sourceFileName() const { return sourceFileName_; }
  RT_API_ATTRS int sourceLine() const { return sourceLine_; }

  RT_API_ATTRS void SetLocation(
      const char *sourceFileName = nullptr, int sourceLine = 0) {
    sourceFileName_ = sourceFileName;
    sourceLine_ = sourceLine;
  }

  // Silence compiler warnings about the format string being
  // non-literal. A more precise control would be
  // __attribute__((format_arg(2))), but it requires the function
  // to return 'char *', which does not work well with noreturn.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wformat-security"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-security"
#endif

  // Device offload compilers do not normally support varargs and va_list,
  // so use C++ variadic templates to forward the crash arguments
  // to regular printf for the device compilation.
  // Try to keep the inline implementations as small as possible.
  template <typename... Args>
  [[noreturn]] RT_API_ATTRS const char *Crash(
      const char *message, Args... args) const {
#if !defined(RT_DEVICE_COMPILATION)
    // Invoke handler set up by the test harness.
    InvokeCrashHandler(message, args...);
#endif
    CrashHeader();
    PrintCrashArgs(message, args...);
    CrashFooter();
  }

  template <typename... Args>
  RT_API_ATTRS void PrintCrashArgs(const char *message, Args... args) const {
#if RT_DEVICE_COMPILATION
    std::printf(message, args...);
#else
    std::fprintf(stderr, message, args...);
#endif
  }

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

  RT_API_ATTRS void CrashHeader() const;
  [[noreturn]] RT_API_ATTRS void CrashFooter() const;
#if !defined(RT_DEVICE_COMPILATION)
  void InvokeCrashHandler(const char *message, ...) const;
  [[noreturn]] void CrashArgs(const char *message, va_list &) const;
#endif
  [[noreturn]] RT_API_ATTRS void CheckFailed(
      const char *predicate, const char *file, int line) const;
  [[noreturn]] RT_API_ATTRS void CheckFailed(const char *predicate) const;

  // For test harnessing - overrides CrashArgs().
  static void RegisterCrashHandler(void (*)(const char *sourceFile,
      int sourceLine, const char *message, va_list &ap));

private:
  const char *sourceFileName_{nullptr};
  int sourceLine_{0};
};

// RUNTIME_CHECK() guarantees evaluation of its predicate.
#define RUNTIME_CHECK(terminator, pred) \
  if (pred) \
    ; \
  else \
    (terminator).CheckFailed(#pred, __FILE__, __LINE__)

#define INTERNAL_CHECK(pred) \
  if (pred) \
    ; \
  else \
    Terminator{__FILE__, __LINE__}.CheckFailed(#pred)

RT_API_ATTRS void NotifyOtherImagesOfNormalEnd();
RT_API_ATTRS void NotifyOtherImagesOfFailImageStatement();
RT_API_ATTRS void NotifyOtherImagesOfErrorTermination();
} // namespace Fortran::runtime

namespace Fortran::runtime::io {
RT_API_ATTRS void FlushOutputOnCrash(const Terminator &);
}

#endif // FORTRAN_RUNTIME_TERMINATOR_H_
