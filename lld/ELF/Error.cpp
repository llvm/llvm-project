//===- Error.cpp ----------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Error.h"
#include "Config.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/raw_ostream.h"

#if !defined(_MSC_VER) && !defined(__MINGW32__)
#include <unistd.h>
#endif

using namespace llvm;

namespace lld {

bool elf::HasError;
raw_ostream *elf::ErrorOS;
StringRef elf::Argv0;

void elf::log(const Twine &Msg) {
  if (Config->Verbose)
    outs() << Argv0 << ": " << Msg << "\n";
}

void elf::warn(const Twine &Msg) {
  if (Config->FatalWarnings)
    error(Msg);
  else
    *ErrorOS << Argv0 << ": warning: " << Msg << "\n";
}

void elf::error(const Twine &Msg) {
  *ErrorOS << Argv0 << ": error: " << Msg << "\n";
  HasError = true;
}

void elf::error(std::error_code EC, const Twine &Prefix) {
  error(Prefix + ": " + EC.message());
}

void elf::exitLld(int Val) {
  // Dealloc/destroy ManagedStatic variables before calling
  // _exit(). In a non-LTO build, this is a nop. In an LTO
  // build allows us to get the output of -time-passes.
  llvm_shutdown();

  outs().flush();
  errs().flush();
  _exit(Val);
}

void elf::fatal(const Twine &Msg) {
  *ErrorOS << Argv0 << ": error: " << Msg << "\n";
  exitLld(1);
}

void elf::fatal(std::error_code EC, const Twine &Prefix) {
  fatal(Prefix + ": " + EC.message());
}

} // namespace lld
