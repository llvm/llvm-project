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

/// Translate clang-cl options before and after offload argument filtering.
class LLVM_LIBRARY_VISIBILITY ClangCLArgs {
  const char *ExpandChar = nullptr;
  bool SupportsForcingFramePointer;

public:
  /// The argument values must outlive this translator.
  ClangCLArgs(const llvm::opt::ArgList &Args, const llvm::Triple &HostTriple);

  /// Append the canonical translation of A to DAL. Return false without
  /// modifying DAL if A is not handled. If supplied, Owner owns synthesized
  /// arguments instead of DAL and must outlive its consumers.
  bool translateArg(llvm::opt::Arg *A, llvm::opt::DerivedArgList &DAL,
                    const llvm::opt::DerivedArgList *Owner = nullptr) const;

  /// Translate a filtered list that may contain new clang-cl options, replacing
  /// earlier /O expansions. The caller owns the returned list; Owner owns its
  /// synthesized arguments and must outlive its consumers.
  static llvm::opt::DerivedArgList *
  translateArgs(const llvm::opt::DerivedArgList &Args,
                const llvm::Triple &HostTriple,
                const llvm::opt::DerivedArgList &Owner);

  /// Normalize MSVC's '#' macro separator, also used by Clang targeting MSVC
  /// and by arguments parsed after host/device filtering. Any replacement
  /// string has the lifetime of Args; otherwise return Value unchanged.
  static const char *translateMacroDefinition(const char *Value,
                                              const llvm::opt::ArgList &Args);
};

} // namespace clang::driver

#endif // LLVM_CLANG_LIB_DRIVER_CLANGCLARGS_H
