//===--- BracesAroundStatement.h - clang-tidy ------- -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides utilities to put braces around a statement.
///
//===----------------------------------------------------------------------===//

#include "clang/AST/Stmt.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"

namespace clang::tidy::utils {

/// A provider of fix-it hints to insert opening and closing braces. An instance
/// of this type is the result of calling `getBraceInsertionsHints` below.
struct BraceInsertionHints {
  /// The position of a potential diagnostic. It coincides with the position of
  /// the opening brace to insert, but can also just be the place to show a
  /// diagnostic in case braces cannot be inserted automatically.
  SourceLocation DiagnosticPos;

  /// Constructor for a no-hint.
  BraceInsertionHints() = default;

  /// Constructor for a valid hint that cannot insert braces automatically.
  BraceInsertionHints(SourceLocation DiagnosticPos)
      : DiagnosticPos(DiagnosticPos) {}

  /// Constructor for a hint offering fix-its for brace insertion. Both
  /// positions must be valid.
  BraceInsertionHints(SourceLocation OpeningBracePos,
                      SourceLocation ClosingBracePos, std::string ClosingBrace)
      : DiagnosticPos(OpeningBracePos), OpeningBracePos(OpeningBracePos),
        ClosingBracePos(ClosingBracePos), ClosingBrace(ClosingBrace) {
    assert(offersFixIts());
  }

  /// Indicates whether the hint provides at least the position of a diagnostic.
  operator bool() const;

  /// Indicates whether the hint provides fix-its to insert braces.
  bool offersFixIts() const;

  /// The number of lines between the inserted opening brace and its closing
  /// counterpart.
  unsigned resultingCompoundLineExtent(const SourceManager &SourceMgr) const;

  /// Fix-it to insert an opening brace.
  FixItHint openingBraceFixIt() const;

  /// Fix-it to insert a closing brace.
  FixItHint closingBraceFixIt() const;

private:
  SourceLocation OpeningBracePos;
  SourceLocation ClosingBracePos;
  std::string ClosingBrace;
};

/// Create fix-it hints for braces that wrap the given statement when applied.
/// The algorithm computing them respects comment before and after the statement
/// and adds line breaks before the braces accordingly.
BraceInsertionHints
getBraceInsertionsHints(const Stmt *const S, const LangOptions &LangOpts,
                        const SourceManager &SM, SourceLocation StartLoc,
                        SourceLocation EndLocHint = SourceLocation());

} // namespace clang::tidy::utils
