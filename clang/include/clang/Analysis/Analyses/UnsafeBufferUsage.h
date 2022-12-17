//===- UnsafeBufferUsage.h - Replace pointers with modern C++ ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines an analysis that aids replacing buffer accesses through
//  raw pointers with safer C++ abstractions such as containers and views/spans.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_ANALYSES_UNSAFEBUFFERUSAGE_H
#define LLVM_CLANG_ANALYSIS_ANALYSES_UNSAFEBUFFERUSAGE_H

#include "clang/AST/Decl.h"
#include "clang/AST/Stmt.h"

namespace clang {

/// The interface that lets the caller handle unsafe buffer usage analysis
/// results by overriding this class's handle... methods.
class UnsafeBufferUsageHandler {
public:
  UnsafeBufferUsageHandler() = default;
  virtual ~UnsafeBufferUsageHandler() = default;

  /// This analyses produces large fixits that are organized into lists
  /// of primitive fixits (individual insertions/removals/replacements).
  using FixItList = llvm::SmallVectorImpl<FixItHint>;

  /// Invoked when an unsafe operation over raw pointers is found.
  virtual void handleUnsafeOperation(const Stmt *Operation) = 0;

  /// Invoked when a fix is suggested against a variable.
  virtual void handleFixableVariable(const VarDecl *Variable,
                                     FixItList &&List) = 0;
};

// This function invokes the analysis and allows the caller to react to it
// through the handler class.
void checkUnsafeBufferUsage(const Decl *D, UnsafeBufferUsageHandler &Handler);

} // end namespace clang

#endif /* LLVM_CLANG_ANALYSIS_ANALYSES_UNSAFEBUFFERUSAGE_H */
