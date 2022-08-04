//===--- PPCachedActions.h - Callbacks for PP cached actions ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Defines the PPCachedActions interface.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LEX_PPCACHEDACTIONS_H
#define LLVM_CLANG_LEX_PPCACHEDACTIONS_H

#include "clang/Basic/SourceLocation.h"

namespace clang {

class Preprocessor;

/// This interface provides a way to override the actions of the preprocessor as
/// it does its thing.
///
/// A client can use this to control how include directives are resolved.
class PPCachedActions {
  virtual void anchor();

public:
  virtual ~PPCachedActions() = default;

  /// \returns the \p FileID that should be used for predefines.
  virtual FileID handlePredefines(Preprocessor &PP) = 0;

  /// \returns the evaluation result for a \p __has_include check.
  virtual bool evaluateHasInclude(Preprocessor &PP, SourceLocation Loc,
                                  bool IsIncludeNext) = 0;

  /// \returns the \p FileID that should be entered for an include directive.
  /// \p None indicates that the directive should be skipped.
  virtual Optional<FileID>
  handleIncludeDirective(Preprocessor &PP, SourceLocation IncludeLoc,
                         SourceLocation AfterDirectiveLoc) = 0;

  /// Notifies the \p PPCachedActions implementation that the preprocessor
  /// finished lexing an include file.
  virtual void exitedFile(Preprocessor &PP, FileID FID) {}
};

} // namespace clang

#endif
