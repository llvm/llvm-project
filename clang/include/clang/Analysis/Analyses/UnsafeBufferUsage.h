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
  virtual void handleUnsafeOperation(const Stmt *Operation,
                                     bool IsRelatedToDecl) = 0;

  /// Invoked when a fix is suggested against a variable.
  virtual void handleFixableVariable(const VarDecl *Variable,
                                     FixItList &&List) = 0;

  /// Returns the text indicating that the user needs to provide input there:
  virtual std::string
  getUserFillPlaceHolder(StringRef HintTextToUser = "placeholder") {
    std::string s = std::string("<# ");
    s += HintTextToUser;
    s += " #>";
    return s;
  }
};

// This function invokes the analysis and allows the caller to react to it
// through the handler class.
void checkUnsafeBufferUsage(const Decl *D, UnsafeBufferUsageHandler &Handler);

namespace internal {
// Tests if any two `FixItHint`s in `FixIts` conflict.  Two `FixItHint`s
// conflict if they have overlapping source ranges.
bool anyConflict(const llvm::SmallVectorImpl<FixItHint> &FixIts,
                 const SourceManager &SM);
} // namespace internal
} // end namespace clang

#endif /* LLVM_CLANG_ANALYSIS_ANALYSES_UNSAFEBUFFERUSAGE_H */
