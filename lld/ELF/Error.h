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

namespace llvm {
class raw_ostream;
}

namespace lld {
namespace elf2 {

extern bool HasError;
extern llvm::raw_ostream *ErrorOS;

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
void fatal(std::error_code EC, const Twine &Prefix);
void fatal(std::error_code EC);

template <typename T> void fatal(const ErrorOr<T> &V, const Twine &Prefix) {
  fatal(V.getError(), Prefix);
}

template <typename T> void fatal(const ErrorOr<T> &V) { fatal(V.getError()); }

} // namespace elf2
} // namespace lld

#endif
