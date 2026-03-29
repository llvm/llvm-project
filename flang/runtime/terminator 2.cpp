//===-- runtime/terminate.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "terminator.h"
#include <cstdio>
#include <cstdlib>

namespace Fortran::runtime {

#if !defined(RT_DEVICE_COMPILATION)
[[maybe_unused]] static void (*crashHandler)(
    const char *, int, const char *, va_list &){nullptr};

void Terminator::RegisterCrashHandler(
    void (*handler)(const char *, int, const char *, va_list &)) {
  crashHandler = handler;
}

void Terminator::InvokeCrashHandler(const char *message, ...) const {
  if (crashHandler) {
    va_list ap;
    va_start(ap, message);
    crashHandler(sourceFileName_, sourceLine_, message, ap);
    va_end(ap);
  }
}

[[noreturn]] void Terminator::CrashArgs(
    const char *message, va_list &ap) const {
  CrashHeader();
  std::vfprintf(stderr, message, ap);
  va_end(ap);
  CrashFooter();
}
#endif

RT_OFFLOAD_API_GROUP_BEGIN

RT_API_ATTRS void Terminator::CrashHeader() const {
#if defined(RT_DEVICE_COMPILATION)
  std::printf("\nfatal Fortran runtime error");
  if (sourceFileName_) {
    std::printf("(%s", sourceFileName_);
    if (sourceLine_) {
      std::printf(":%d", sourceLine_);
    }
    std::printf(")");
  }
  std::printf(": ");
#else
  std::fputs("\nfatal Fortran runtime error", stderr);
  if (sourceFileName_) {
    std::fprintf(stderr, "(%s", sourceFileName_);
    if (sourceLine_) {
      std::fprintf(stderr, ":%d", sourceLine_);
    }
    fputc(')', stderr);
  }
  std::fputs(": ", stderr);
#endif
}

[[noreturn]] RT_API_ATTRS void Terminator::CrashFooter() const {
#if defined(RT_DEVICE_COMPILATION)
  std::printf("\n");
#else
  fputc('\n', stderr);
  // FIXME: re-enable the flush along with the IO enabling.
  io::FlushOutputOnCrash(*this);
#endif
  NotifyOtherImagesOfErrorTermination();
#if defined(RT_DEVICE_COMPILATION)
#if defined(__CUDACC__)
  // NVCC supports __trap().
  __trap();
#elif defined(__clang__)
  // Clang supports __builtin_trap().
  __builtin_trap();
#else
#error "unsupported compiler"
#endif
#else
  std::abort();
#endif
}

[[noreturn]] RT_API_ATTRS void Terminator::CheckFailed(
    const char *predicate, const char *file, int line) const {
  Crash("Internal error: RUNTIME_CHECK(%s) failed at %s(%d)", predicate, file,
      line);
}

[[noreturn]] RT_API_ATTRS void Terminator::CheckFailed(
    const char *predicate) const {
  Crash("Internal error: RUNTIME_CHECK(%s) failed at %s(%d)", predicate,
      sourceFileName_, sourceLine_);
}

// TODO: These will be defined in the coarray runtime library
RT_API_ATTRS void NotifyOtherImagesOfNormalEnd() {}
RT_API_ATTRS void NotifyOtherImagesOfFailImageStatement() {}
RT_API_ATTRS void NotifyOtherImagesOfErrorTermination() {}

RT_OFFLOAD_API_GROUP_END

} // namespace Fortran::runtime
