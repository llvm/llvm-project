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
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

#if !defined(_MSC_VER) && !defined(__MINGW32__)
#include <unistd.h>
#endif

using namespace llvm;

namespace lld {
namespace coff {

void fatal(const Twine &Msg) {
  if (sys::Process::StandardErrHasColors()) {
    errs().changeColor(raw_ostream::RED, /*bold=*/true);
    errs() << "error: ";
    errs().resetColor();
  } else {
    errs() << "error: ";
  }
  errs() << Msg << "\n";

  outs().flush();
  errs().flush();
  _exit(1);
}

void fatal(std::error_code EC, const Twine &Msg) {
  fatal(Msg + ": " + EC.message());
}

void fatal(llvm::Error &Err, const Twine &Msg) {
  fatal(errorToErrorCode(std::move(Err)), Msg);
}

} // namespace coff
} // namespace lld
