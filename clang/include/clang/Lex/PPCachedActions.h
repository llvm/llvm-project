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
#include "llvm/ADT/SmallVector.h"

namespace clang {

class IdentifierInfo;
class Module;
class Preprocessor;

/// This interface provides a way to override the actions of the preprocessor as
/// it does its thing.
///
/// A client can use this to control how include directives are resolved.
class PPCachedActions {
  virtual void anchor();

public:
  /// The file that is included by an \c #include directive.
  struct IncludeFile {
    FileID FID;
    Module *Submodule;
  };
  /// The module that is imported by an \c #include directive or \c \@import.
  struct IncludeModule {
    SmallVector<std::pair<IdentifierInfo *, SourceLocation>, 2> ImportPath;
    // Whether this module should only be "marked visible" rather than imported.
    bool VisibilityOnly;
  };

  virtual ~PPCachedActions() = default;

  /// \returns the \p FileID that should be used for predefines.
  virtual FileID handlePredefines(Preprocessor &PP) = 0;

  /// \returns the evaluation result for a \p __has_include check.
  virtual bool evaluateHasInclude(Preprocessor &PP, SourceLocation Loc,
                                  bool IsIncludeNext) = 0;

  /// \returns the file that should be entered or module that should be imported
  /// for an \c #include directive. \c {} indicates that the directive
  /// should be skipped.
  virtual std::variant<std::monostate, IncludeFile, IncludeModule>
  handleIncludeDirective(Preprocessor &PP, SourceLocation IncludeLoc,
                         SourceLocation AfterDirectiveLoc) = 0;

  /// Notifies the \p PPCachedActions implementation that the preprocessor
  /// finished lexing an include file.
  virtual void exitedFile(Preprocessor &PP, FileID FID) {}
};

} // namespace clang

#endif
