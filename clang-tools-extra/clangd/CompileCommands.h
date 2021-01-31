//===--- CompileCommands.h - Manipulation of compile flags -------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_COMPILECOMMANDS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_COMPILECOMMANDS_H

#include "support/Threading.h"
#include "clang/Tooling/ArgumentsAdjusters.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "llvm/ADT/StringMap.h"
#include <deque>
#include <string>
#include <vector>

namespace clang {
namespace clangd {

// CommandMangler transforms compile commands from some external source
// for use in clangd. This means:
//  - running the frontend only, stripping args regarding output files etc
//  - forcing the use of clangd's builtin headers rather than clang's
//  - resolving argv0 as cc1 expects
//  - injecting -isysroot flags on mac as the system clang does
struct CommandMangler {
  // Absolute path to clang.
  llvm::Optional<std::string> ClangPath;
  // Directory containing builtin headers.
  llvm::Optional<std::string> ResourceDir;
  // Root for searching for standard library (passed to -isysroot).
  llvm::Optional<std::string> Sysroot;

  // A command-mangler that doesn't know anything about the system.
  // This is hermetic for unit-tests, but won't work well in production.
  static CommandMangler forTests();
  // Probe the system and build a command-mangler that knows the toolchain.
  //  - try to find clang on $PATH, otherwise fake a path near clangd
  //  - find the resource directory installed near clangd
  //  - on mac, find clang and isysroot by querying the `xcrun` launcher
  static CommandMangler detect();

  void adjust(std::vector<std::string> &Cmd) const;
  explicit operator clang::tooling::ArgumentsAdjuster() &&;

private:
  CommandMangler() = default;
  Memoize<llvm::StringMap<std::string>> ResolvedDrivers;
  Memoize<llvm::StringMap<std::string>> ResolvedDriversNoFollow;
};

// Removes args from a command-line in a semantically-aware way.
//
// Internally this builds a large (0.5MB) table of clang options on first use.
// Both strip() and process() are fairly cheap after that.
//
// FIXME: this reimplements much of OptTable, it might be nice to expose more.
// The table-building strategy may not make sense outside clangd.
class ArgStripper {
public:
  ArgStripper() = default;
  ArgStripper(ArgStripper &&) = default;
  ArgStripper(const ArgStripper &) = delete;
  ArgStripper &operator=(ArgStripper &&) = default;
  ArgStripper &operator=(const ArgStripper &) = delete;

  // Adds the arg to the set which should be removed.
  //
  // Recognized clang flags are stripped semantically. When "-I" is stripped:
  //  - so is its value (either as -Ifoo or -I foo)
  //  - aliases like --include-directory=foo are also stripped
  //  - CL-style /Ifoo will be removed if the args indicate MS-compatible mode
  // Compile args not recognized as flags are removed literally, except:
  //  - strip("ABC*") will remove any arg with an ABC prefix.
  //
  // In either case, the -Xclang prefix will be dropped if present.
  void strip(llvm::StringRef Arg);
  // Remove the targets from a compile command, in-place.
  void process(std::vector<std::string> &Args) const;

private:
  // Deletion rules, to be checked for each arg.
  struct Rule {
    llvm::StringRef Text;    // Rule applies only if arg begins with Text.
    unsigned char Modes = 0; // Rule applies only in specified driver modes.
    uint16_t Priority = 0;   // Lower is better.
    uint16_t ExactArgs = 0;  // Num args consumed when Arg == Text.
    uint16_t PrefixArgs = 0; // Num args consumed when Arg starts with Text.
  };
  static llvm::ArrayRef<Rule> rulesFor(llvm::StringRef Arg);
  const Rule *matchingRule(llvm::StringRef Arg, unsigned Mode,
                           unsigned &ArgCount) const;
  llvm::SmallVector<Rule> Rules;
  std::deque<std::string> Storage; // Store strings not found in option table.
};

} // namespace clangd
} // namespace clang

#endif
