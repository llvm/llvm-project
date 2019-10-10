//===--- SourceCode.h - Source code manipulation routines -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file provides functions that simplify extraction of source code.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_TRANSFORMER_SOURCE_CODE_H
#define LLVM_CLANG_TOOLING_TRANSFORMER_SOURCE_CODE_H

#include "clang/AST/ASTContext.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/TokenKinds.h"

namespace clang {
namespace tooling {

/// Extends \p Range to include the token \p Next, if it immediately follows the
/// end of the range. Otherwise, returns \p Range unchanged.
CharSourceRange maybeExtendRange(CharSourceRange Range, tok::TokenKind Next,
                                 ASTContext &Context);

/// Returns the source range spanning the node, extended to include \p Next, if
/// it immediately follows \p Node. Otherwise, returns the normal range of \p
/// Node.  See comments on `getExtendedText()` for examples.
template <typename T>
CharSourceRange getExtendedRange(const T &Node, tok::TokenKind Next,
                                 ASTContext &Context) {
  return maybeExtendRange(CharSourceRange::getTokenRange(Node.getSourceRange()),
                          Next, Context);
}

/// Returns the source-code text in the specified range.
StringRef getText(CharSourceRange Range, const ASTContext &Context);

/// Returns the source-code text corresponding to \p Node.
template <typename T>
StringRef getText(const T &Node, const ASTContext &Context) {
  return getText(CharSourceRange::getTokenRange(Node.getSourceRange()),
                 Context);
}

/// Returns the source text of the node, extended to include \p Next, if it
/// immediately follows the node. Otherwise, returns the text of just \p Node.
///
/// For example, given statements S1 and S2 below:
/// \code
///   {
///     // S1:
///     if (!x) return foo();
///     // S2:
///     if (!x) { return 3; }
///   }
/// \endcode
/// then
/// \code
///   getText(S1, Context) = "if (!x) return foo()"
///   getExtendedText(S1, tok::TokenKind::semi, Context)
///     = "if (!x) return foo();"
///   getExtendedText(*S1.getThen(), tok::TokenKind::semi, Context)
///     = "return foo();"
///   getExtendedText(*S2.getThen(), tok::TokenKind::semi, Context)
///     = getText(S2, Context) = "{ return 3; }"
/// \endcode
template <typename T>
StringRef getExtendedText(const T &Node, tok::TokenKind Next,
                          ASTContext &Context) {
  return getText(getExtendedRange(Node, Next, Context), Context);
}

// Attempts to resolve the given range to one that can be edited by a rewrite;
// generally, one that starts and ends within a particular file. It supports
// a limited set of cases involving source locations in macro expansions.
llvm::Optional<CharSourceRange>
getRangeForEdit(const CharSourceRange &EditRange, const SourceManager &SM,
                const LangOptions &LangOpts);

inline llvm::Optional<CharSourceRange>
getRangeForEdit(const CharSourceRange &EditRange, const ASTContext &Context) {
  return getRangeForEdit(EditRange, Context.getSourceManager(),
                         Context.getLangOpts());
}
} // namespace tooling
} // namespace clang
#endif // LLVM_CLANG_TOOLING_TRANSFORMER_SOURCE_CODE_H
