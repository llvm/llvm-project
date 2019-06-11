//===- tools/lld/lld.cpp - Linker Driver Dispatcher -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the main function of the lld executable. The main
// function is a thin wrapper which dispatches to the platform specific
// driver.
//
// lld is a single executable that contains four different linkers for ELF,
// COFF, WebAssembly and Mach-O. The main function dispatches according to
// argv[0] (i.e. command name). The most common name for each target is shown
// below:
//
//  - ld.lld:    ELF (Unix)
//  - ld64:      Mach-O (macOS)
//  - lld-link:  COFF (Windows)
//  - ld-wasm:   WebAssembly
//
// lld can be invoked as "lld" along with "-flavor" option. This is for
// backward compatibility and not recommended.
//
//===----------------------------------------------------------------------===//

#include "lld/Common/Driver.h"
#include "lld/Common/Memory.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include <cstdlib>

using namespace lld;
using namespace llvm;
using namespace llvm::sys;

enum Flavor {
  Invalid,
  Gnu,     // -flavor gnu
  WinLink, // -flavor link
  Darwin,  // -flavor darwin
  Wasm,    // -flavor wasm
};

LLVM_ATTRIBUTE_NORETURN static void die(const Twine &S) {
  errs() << S << "\n";
  exit(1);
}

static Flavor getFlavor(StringRef S) {
  return StringSwitch<Flavor>(S)
      .CasesLower("ld", "ld.lld", "gnu", Gnu)
      .CasesLower("wasm", "ld-wasm", Wasm)
      .CaseLower("link", WinLink)
      .CasesLower("ld64", "ld64.lld", "darwin", Darwin)
      .Default(Invalid);
}

static cl::TokenizerCallback getDefaultQuotingStyle() {
  if (Triple(sys::getProcessTriple()).getOS() == Triple::Win32)
    return cl::TokenizeWindowsCommandLine;
  return cl::TokenizeGNUCommandLine;
}

static bool isPETargetName(StringRef S) {
  return S == "i386pe" || S == "i386pep" || S == "thumb2pe" || S == "arm64pe";
}

static bool isPETarget(std::vector<const char *> &V) {
  for (auto It = V.begin(); It + 1 != V.end(); ++It) {
    if (StringRef(*It) != "-m")
      continue;
    return isPETargetName(*(It + 1));
  }
  // Expand response files (arguments in the form of @<filename>)
  // to allow detecting the -m argument from arguments in them.
  SmallVector<const char *, 256> ExpandedArgs(V.data(), V.data() + V.size());
  cl::ExpandResponseFiles(Saver, getDefaultQuotingStyle(), ExpandedArgs);
  for (auto It = ExpandedArgs.begin(); It + 1 != ExpandedArgs.end(); ++It) {
    if (StringRef(*It) != "-m")
      continue;
    return isPETargetName(*(It + 1));
  }
  return false;
}

static Flavor parseProgname(StringRef Progname) {
#if __APPLE__
  // Use Darwin driver for "ld" on Darwin.
  if (Progname == "ld")
    return Darwin;
#endif

#if LLVM_ON_UNIX
  // Use GNU driver for "ld" on other Unix-like system.
  if (Progname == "ld")
    return Gnu;
#endif

  // Progname may be something like "lld-gnu". Parse it.
  SmallVector<StringRef, 3> V;
  Progname.split(V, "-");
  for (StringRef S : V)
    if (Flavor F = getFlavor(S))
      return F;
  return Invalid;
}

static Flavor parseFlavor(std::vector<const char *> &V) {
  // Parse -flavor option.
  if (V.size() > 1 && V[1] == StringRef("-flavor")) {
    if (V.size() <= 2)
      die("missing arg value for '-flavor'");
    Flavor F = getFlavor(V[2]);
    if (F == Invalid)
      die("Unknown flavor: " + StringRef(V[2]));
    V.erase(V.begin() + 1, V.begin() + 3);
    return F;
  }

  // Deduct the flavor from argv[0].
  StringRef Arg0 = path::filename(V[0]);
  if (Arg0.endswith_lower(".exe"))
    Arg0 = Arg0.drop_back(4);
  return parseProgname(Arg0);
}

// If this function returns true, lld calls _exit() so that it quickly
// exits without invoking destructors of globally allocated objects.
//
// We don't want to do that if we are running tests though, because
// doing that breaks leak sanitizer. So, lit sets this environment variable,
// and we use it to detect whether we are running tests or not.
static bool canExitEarly() { return StringRef(getenv("LLD_IN_TEST")) != "1"; }

/// Universal linker main(). This linker emulates the gnu, darwin, or
/// windows linker based on the argv[0] or -flavor option.
int main(int Argc, const char **Argv) {
  InitLLVM X(Argc, Argv);

  std::vector<const char *> Args(Argv, Argv + Argc);
  switch (parseFlavor(Args)) {
  case Gnu:
    if (isPETarget(Args))
      return !mingw::link(Args);
    return !elf::link(Args, canExitEarly());
  case WinLink:
    return !coff::link(Args, canExitEarly());
  case Darwin:
    return !mach_o::link(Args, canExitEarly());
  case Wasm:
    return !wasm::link(Args, canExitEarly());
  default:
    die("lld is a generic driver.\n"
        "Invoke ld.lld (Unix), ld64.lld (macOS), lld-link (Windows), wasm-ld"
        " (WebAssembly) instead");
  }
}
