//===- Error.h --------------------------------------------------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_COFF_ERROR_H
#define LLD_COFF_ERROR_H

#include "lld/Core/LLVM.h"

namespace lld {
namespace elf {

extern bool HasError;
extern llvm::raw_ostream *ErrorOS;

void log(const Twine &Msg);
void warning(const Twine &Msg);

void error(const Twine &Msg);
bool error(std::error_code EC, const Twine &Prefix);
bool error(std::error_code EC);

template <typename T> bool error(const ErrorOr<T> &V, const Twine &Prefix) {
  return error(V.getError(), Prefix);
}

template <typename T> bool error(const ErrorOr<T> &V) {
  return error(V.getError());
}

LLVM_ATTRIBUTE_NORETURN void fatal(const Twine &Msg);
LLVM_ATTRIBUTE_NORETURN void fatal(const Twine &Msg, const Twine &Prefix);
void fatal(std::error_code EC);

template <class T> T fatal(ErrorOr<T> EO) {
  if (EO)
    return std::move(*EO);
  fatal(EO.getError().message());
}

template <class T> T fatal(ErrorOr<T> EO, const Twine &Prefix) {
  if (EO)
    return std::move(*EO);
  fatal(EO.getError().message(), Prefix);
}

} // namespace elf
} // namespace lld

#endif
