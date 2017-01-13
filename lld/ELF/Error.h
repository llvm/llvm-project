//===- Error.h --------------------------------------------------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// In LLD, we have three levels of errors: fatal, error or warn.
//
// Fatal makes the program exit immediately with an error message.
// You shouldn't use it except for reporting a corrupted input file.
//
// Error prints out an error message and increment a global variable
// ErrorCount to record the fact that we met an error condition. It does
// not exit, so it is safe for a lld-as-a-library use case. It is generally
// useful because it can report more than one errors in a single run.
//
// Warn doesn't do anything but printing out a given message.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_ERROR_H
#define LLD_ELF_ERROR_H

#include "lld/Core/LLVM.h"

#include "llvm/Support/Error.h"

namespace lld {
namespace elf {

extern uint64_t ErrorCount;
extern llvm::raw_ostream *ErrorOS;
extern llvm::StringRef Argv0;

void log(const Twine &Msg);
void warn(const Twine &Msg);
void error(const Twine &Msg);
LLVM_ATTRIBUTE_NORETURN void fatal(const Twine &Msg);

LLVM_ATTRIBUTE_NORETURN void exitLld(int Val);

// check() functions are convenient functions to strip errors
// from error-or-value objects.
template <class T> T check(ErrorOr<T> E) {
  if (auto EC = E.getError())
    fatal(EC.message());
  return std::move(*E);
}

template <class T> T check(Expected<T> E) {
  if (!E)
    fatal(llvm::toString(E.takeError()));
  return std::move(*E);
}

template <class T> T check(ErrorOr<T> E, const Twine &Prefix) {
  if (auto EC = E.getError())
    fatal(Prefix + ": " + EC.message());
  return std::move(*E);
}

template <class T> T check(Expected<T> E, const Twine &Prefix) {
  if (!E)
    fatal(Prefix + ": " + errorToErrorCode(E.takeError()).message());
  return std::move(*E);
}

} // namespace elf
} // namespace lld

#endif
