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
#include "lld/Common/ErrorHandler.h"
#include "lld/Common/Memory.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/CrashRecoveryContext.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PluginLoader.h"
#include <cstdlib>

using namespace lld;
using namespace llvm;
using namespace llvm::sys;

enum Flavor {
  Invalid,
  Gnu,       // -flavor gnu
  WinLink,   // -flavor link
  Darwin,    // -flavor darwin
  DarwinOld, // -flavor darwinold
  Wasm,      // -flavor wasm
};

[[noreturn]] static void die(const Twine &s) {
  llvm::errs() << s << "\n";
  exit(1);
}

static Flavor getFlavor(StringRef s) {
  return StringSwitch<Flavor>(s)
      .CasesLower("ld", "ld.lld", "gnu", Gnu)
      .CasesLower("wasm", "ld-wasm", Wasm)
      .CaseLower("link", WinLink)
      .CasesLower("ld64", "ld64.lld", "darwin", "darwinnew",
                  "ld64.lld.darwinnew", Darwin)
      .CasesLower("darwinold", "ld64.lld.darwinold", DarwinOld)
      .Default(Invalid);
}

static cl::TokenizerCallback getDefaultQuotingStyle() {
  if (Triple(sys::getProcessTriple()).getOS() == Triple::Win32)
    return cl::TokenizeWindowsCommandLine;
  return cl::TokenizeGNUCommandLine;
}

static bool isPETargetName(StringRef s) {
  return s == "i386pe" || s == "i386pep" || s == "thumb2pe" || s == "arm64pe";
}

static bool isPETarget(std::vector<const char *> &v) {
  for (auto it = v.begin(); it + 1 != v.end(); ++it) {
    if (StringRef(*it) != "-m")
      continue;
    return isPETargetName(*(it + 1));
  }
  // Expand response files (arguments in the form of @<filename>)
  // to allow detecting the -m argument from arguments in them.
  SmallVector<const char *, 256> expandedArgs(v.data(), v.data() + v.size());
  cl::ExpandResponseFiles(saver, getDefaultQuotingStyle(), expandedArgs);
  for (auto it = expandedArgs.begin(); it + 1 != expandedArgs.end(); ++it) {
    if (StringRef(*it) != "-m")
      continue;
    return isPETargetName(*(it + 1));
  }

#ifdef LLD_DEFAULT_LD_LLD_IS_MINGW
  return true;
#else
  return false;
#endif
}

static Flavor parseProgname(StringRef progname) {
  // Use GNU driver for "ld" by default.
  if (progname == "ld")
    return Gnu;

  // Progname may be something like "lld-gnu". Parse it.
  SmallVector<StringRef, 3> v;
  progname.split(v, "-");
  for (StringRef s : v)
    if (Flavor f = getFlavor(s))
      return f;
  return Invalid;
}

static Flavor parseFlavor(std::vector<const char *> &v) {
  // Parse -flavor option.
  if (v.size() > 1 && v[1] == StringRef("-flavor")) {
    if (v.size() <= 2)
      die("missing arg value for '-flavor'");
    Flavor f = getFlavor(v[2]);
    if (f == Invalid)
      die("Unknown flavor: " + StringRef(v[2]));
    v.erase(v.begin() + 1, v.begin() + 3);
    return f;
  }

  // Deduct the flavor from argv[0].
  StringRef arg0 = path::filename(v[0]);
  if (arg0.endswith_insensitive(".exe"))
    arg0 = arg0.drop_back(4);
  return parseProgname(arg0);
}

/// Universal linker main(). This linker emulates the gnu, darwin, or
/// windows linker based on the argv[0] or -flavor option.
static int lldMain(int argc, const char **argv, llvm::raw_ostream &stdoutOS,
                   llvm::raw_ostream &stderrOS, bool exitEarly = true) {
  std::vector<const char *> args(argv, argv + argc);
  switch (parseFlavor(args)) {
  case Gnu:
    if (isPETarget(args))
      return !mingw::link(args, exitEarly, stdoutOS, stderrOS);
    return !elf::link(args, exitEarly, stdoutOS, stderrOS);
  case WinLink:
    return !coff::link(args, exitEarly, stdoutOS, stderrOS);
  case Darwin:
    return !macho::link(args, exitEarly, stdoutOS, stderrOS);
  case DarwinOld:
    return !mach_o::link(args, exitEarly, stdoutOS, stderrOS);
  case Wasm:
    return !lld::wasm::link(args, exitEarly, stdoutOS, stderrOS);
  default:
    die("lld is a generic driver.\n"
        "Invoke ld.lld (Unix), ld64.lld (macOS), lld-link (Windows), wasm-ld"
        " (WebAssembly) instead");
  }
}

// Similar to lldMain except that exceptions are caught.
SafeReturn lld::safeLldMain(int argc, const char **argv,
                            llvm::raw_ostream &stdoutOS,
                            llvm::raw_ostream &stderrOS) {
  int r = 0;
  {
    // The crash recovery is here only to be able to recover from arbitrary
    // control flow when fatal() is called (through setjmp/longjmp or
    // __try/__except).
    llvm::CrashRecoveryContext crc;
    if (!crc.RunSafely([&]() {
          r = lldMain(argc, argv, stdoutOS, stderrOS, /*exitEarly=*/false);
        }))
      return {crc.RetCode, /*canRunAgain=*/false};
  }

  // Cleanup memory and reset everything back in pristine condition. This path
  // is only taken when LLD is in test, or when it is used as a library.
  llvm::CrashRecoveryContext crc;
  if (!crc.RunSafely([&]() { errorHandler().reset(); })) {
    // The memory is corrupted beyond any possible recovery.
    return {r, /*canRunAgain=*/false};
  }
  return {r, /*canRunAgain=*/true};
}

// When in lit tests, tells how many times the LLD tool should re-execute the
// main loop with the same inputs. When not in test, returns a value of 0 which
// signifies that LLD shall not release any memory after execution, to speed up
// process destruction.
static unsigned inTestVerbosity() {
  unsigned v = 0;
  StringRef(getenv("LLD_IN_TEST")).getAsInteger(10, v);
  return v;
}

int main(int argc, const char **argv) {
  InitLLVM x(argc, argv);

  // Not running in lit tests, just take the shortest codepath with global
  // exception handling and no memory cleanup on exit.
  if (!inTestVerbosity())
    return lldMain(argc, argv, llvm::outs(), llvm::errs());

  Optional<int> mainRet;
  CrashRecoveryContext::Enable();

  for (unsigned i = inTestVerbosity(); i > 0; --i) {
    // Disable stdout/stderr for all iterations but the last one.
    if (i != 1)
      errorHandler().disableOutput = true;

    // Execute one iteration.
    auto r = safeLldMain(argc, argv, llvm::outs(), llvm::errs());
    if (!r.canRunAgain)
      exitLld(r.ret); // Exit now, can't re-execute again.

    if (!mainRet) {
      mainRet = r.ret;
    } else if (r.ret != *mainRet) {
      // Exit now, to fail the tests if the result is different between runs.
      return r.ret;
    }
  }
  return *mainRet;
}
