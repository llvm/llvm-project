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

#include "entry-names.h"
#include <cstdarg>

namespace Fortran::runtime {

// A mixin class for statement-specific image error termination
// for errors detected in the runtime library
class Terminator {
public:
  Terminator() {}
  Terminator(const Terminator &) = default;
  explicit Terminator(const char *sourceFileName, int sourceLine = 0)
      : sourceFileName_{sourceFileName}, sourceLine_{sourceLine} {}
  void SetLocation(const char *sourceFileName = nullptr, int sourceLine = 0) {
    sourceFileName_ = sourceFileName;
    sourceLine_ = sourceLine;
  }
  [[noreturn]] void Crash(const char *message, ...) const;
  [[noreturn]] void CrashArgs(const char *message, va_list &) const;
  [[noreturn]] void CheckFailed(
      const char *predicate, const char *file, int line) const;

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

void NotifyOtherImagesOfNormalEnd();
void NotifyOtherImagesOfFailImageStatement();
void NotifyOtherImagesOfErrorTermination();
} // namespace Fortran::runtime

namespace Fortran::runtime::io {
void FlushOutputOnCrash(const Terminator &);
}

#endif // FORTRAN_RUNTIME_TERMINATOR_H_
