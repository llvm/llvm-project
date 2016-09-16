//===- Error.h --------------------------------------------------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_ERROR_H
#define LLD_ELF_ERROR_H

#include "lld/Core/LLVM.h"

namespace lld {
namespace elf {

extern bool HasError;
extern llvm::raw_ostream *ErrorOS;

void log(const Twine &Msg);
void warning(const Twine &Msg);

void error(const Twine &Msg);
void error(std::error_code EC, const Twine &Prefix);

template <typename T> void error(const ErrorOr<T> &V, const Twine &Prefix) {
  error(V.getError(), Prefix);
}

LLVM_ATTRIBUTE_NORETURN void fatal(const Twine &Msg);
LLVM_ATTRIBUTE_NORETURN void fatal(const Twine &Msg, const Twine &Prefix);

template <class T> T check(ErrorOr<T> E) {
  if (auto EC = E.getError())
    fatal(EC.message());
  return std::move(*E);
}

template <class T> T check(Expected<T> E) {
  if (!E)
    fatal(errorToErrorCode(E.takeError()).message());
  return std::move(*E);
}

template <class T> T check(ErrorOr<T> E, const Twine &Prefix) {
  if (auto EC = E.getError())
    fatal(EC.message(), Prefix);
  return std::move(*E);
}

template <class T> T check(Expected<T> E, const Twine &Prefix) {
  if (!E)
    fatal(errorToErrorCode(E.takeError()).message(), Prefix);
  return std::move(*E);
}

} // namespace elf
} // namespace lld

#endif
