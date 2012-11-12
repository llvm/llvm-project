//===--- ArgumentsAdjusters.cpp - Command line arguments adjuster ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions of classes which implement ArgumentsAdjuster
// interface.
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/ArgumentsAdjusters.h"

namespace clang {
namespace tooling {

void ArgumentsAdjuster::anchor() {
}

/// Add -fsyntax-only option to the commnand line arguments.
CommandLineArguments
ClangSyntaxOnlyAdjuster::Adjust(const CommandLineArguments &Args) {
  CommandLineArguments AdjustedArgs = Args;
  // FIXME: Remove options that generate output.
  AdjustedArgs.push_back("-fsyntax-only");
  return AdjustedArgs;
}

} // end namespace tooling
} // end namespace clang

