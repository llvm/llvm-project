//===- CrossTUAnalysisHelper.h - Abstraction layer for CTU ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_ANALYSIS_CROSS_TU_HELPER_H
#define LLVM_CLANG_ANALYSIS_CROSS_TU_HELPER_H

#include "llvm/ADT/Optional.h"
#include "clang/Basic/SourceManager.h"

namespace clang {

class ASTUnit;

/// This class is an abstract interface acting as a bridge between
/// an analysis that requires lookups across translation units (a user
/// of that interface) and the facility that implements such lookups
/// (an implementation of that interface). This is useful to break direct
/// link-time dependencies between the (possibly shared) libraries in which
/// the user and the implementation live.
class CrossTUAnalysisHelper {
public:
  /// Determine the original source location in the original TU for an
  /// imported source location.
  /// \p ToLoc Source location in the imported-to AST.
  /// \return Source location in the imported-from AST and the corresponding
  /// ASTUnit object (the AST was loaded from a file using an internal ASTUnit
  /// object that is returned here).
  /// If any error happens (ToLoc is a non-imported source location) empty is
  /// returned.
  virtual llvm::Optional<std::pair<SourceLocation /*FromLoc*/, ASTUnit *>>
  getImportedFromSourceLocation(SourceLocation ToLoc) const = 0;

  virtual ~CrossTUAnalysisHelper() {}
};
} // namespace clang

#endif // LLVM_CLANG_ANALYSIS_CROSS_TU_HELPER_H
