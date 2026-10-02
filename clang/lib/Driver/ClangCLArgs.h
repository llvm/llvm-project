//===--- ClangCLArgs.h - clang-cl arguments ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_DRIVER_CLANGCLARGS_H
#define LLVM_CLANG_LIB_DRIVER_CLANGCLARGS_H

#include "llvm/Support/Compiler.h"

namespace llvm {
class Triple;
namespace opt {
class Arg;
class ArgList;
class DerivedArgList;
} // namespace opt
} // namespace llvm

namespace clang::driver {

/// Translate clang-cl options before host and device arguments are split.
class LLVM_LIBRARY_VISIBILITY ClangCLArgs {
  const char *ExpandChar = nullptr;
  bool SupportsForcingFramePointer;

public:
  /// The argument values must outlive this translator.
  ClangCLArgs(const llvm::opt::ArgList &Args, const llvm::Triple &HostTriple);

  /// Append A and its canonical expansion to DAL. Return false without
  /// modifying DAL if A is not handled.
  bool translateArg(llvm::opt::Arg *A, llvm::opt::DerivedArgList &DAL) const;
};

} // namespace clang::driver

#endif // LLVM_CLANG_LIB_DRIVER_CLANGCLARGS_H
