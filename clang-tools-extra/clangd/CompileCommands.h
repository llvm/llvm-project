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

} // namespace clangd
} // namespace clang

#endif
