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

void warning(const Twine &Msg) { llvm::errs() << Msg << "\n"; }

void error(const Twine &Msg) {
  llvm::errs() << Msg << "\n";
  HasError = true;
}

void error(std::error_code EC, const Twine &Prefix) {
  if (EC)
    error(Prefix + ": " + EC.message());
}

void error(std::error_code EC) {
  if (EC)
    error(EC.message());
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
