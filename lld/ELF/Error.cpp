//===- Error.cpp ----------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Error.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

namespace lld {
namespace elf2 {

bool HasError;
llvm::raw_ostream *ErrorOS;

void warning(const Twine &Msg) { llvm::errs() << Msg << "\n"; }

void error(const Twine &Msg) {
  *ErrorOS << Msg << "\n";
  HasError = true;
}

bool error(std::error_code EC, const Twine &Prefix) {
  if (!EC)
    return false;
  error(Prefix + ": " + EC.message());
  return true;
}

bool error(std::error_code EC) {
  if (!EC)
    return false;
  error(EC.message());
  return true;
}

void fatal(const Twine &Msg) {
  llvm::errs() << Msg << "\n";
  exit(1);
}

void fatal(std::error_code EC, const Twine &Prefix) {
  if (EC)
    fatal(Prefix + ": " + EC.message());
}

void fatal(std::error_code EC) {
  if (EC)
    fatal(EC.message());
}

} // namespace elf2
} // namespace lld
