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
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

namespace lld {
namespace coff {

void fatal(const Twine &Msg) {
  llvm::errs() << Msg << "\n";
  exit(1);
}

void fatal(std::error_code EC, const Twine &Msg) {
  fatal(Msg + ": " + EC.message());
}

void fatal(llvm::Error &Err, const Twine &Msg) {
  fatal(errorToErrorCode(std::move(Err)), Msg);
}

} // namespace coff
} // namespace lld
