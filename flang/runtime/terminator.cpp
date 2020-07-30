//===-- runtime/terminate.cpp -----------------------------------*- C++ -*-===//
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

[[noreturn]] void Terminator::Crash(const char *message, ...) const {
  va_list ap;
  va_start(ap, message);
  CrashArgs(message, ap);
}

static void (*crashHandler)(const char *, int, const char *, va_list &){
    nullptr};

void Terminator::RegisterCrashHandler(
    void (*handler)(const char *, int, const char *, va_list &)) {
  crashHandler = handler;
}

[[noreturn]] void Terminator::CrashArgs(
    const char *message, va_list &ap) const {
  if (crashHandler) {
    crashHandler(sourceFileName_, sourceLine_, message, ap);
  }
  std::fputs("\nfatal Fortran runtime error", stderr);
  if (sourceFileName_) {
    std::fprintf(stderr, "(%s", sourceFileName_);
    if (sourceLine_) {
      std::fprintf(stderr, ":%d", sourceLine_);
    }
    fputc(')', stderr);
  }
  std::fputs(": ", stderr);
  std::vfprintf(stderr, message, ap);
  fputc('\n', stderr);
  va_end(ap);
  io::FlushOutputOnCrash(*this);
  NotifyOtherImagesOfErrorTermination();
  std::abort();
}

[[noreturn]] void Terminator::CheckFailed(
    const char *predicate, const char *file, int line) const {
  Crash("Internal error: RUNTIME_CHECK(%s) failed at %s(%d)", predicate, file,
      line);
}

// TODO: These will be defined in the coarray runtime library
void NotifyOtherImagesOfNormalEnd() {}
void NotifyOtherImagesOfFailImageStatement() {}
void NotifyOtherImagesOfErrorTermination() {}
} // namespace Fortran::runtime
