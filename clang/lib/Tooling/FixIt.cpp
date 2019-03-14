//===--- FixIt.cpp - FixIt Hint utilities -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains implementations of utitilies to ease source code rewriting
// by providing helper functions related to FixItHint.
//
//===----------------------------------------------------------------------===//
#include "clang/Tooling/FixIt.h"
#include "clang/Lex/Lexer.h"

namespace clang {
namespace tooling {
namespace fixit {

namespace internal {
StringRef getText(CharSourceRange Range, const ASTContext &Context) {
  return Lexer::getSourceText(Range, Context.getSourceManager(),
                              Context.getLangOpts());
}

CharSourceRange maybeExtendRange(CharSourceRange Range, tok::TokenKind Next,
                                 ASTContext &Context) {
  Optional<Token> Tok = Lexer::findNextToken(
      Range.getEnd(), Context.getSourceManager(), Context.getLangOpts());
  if (!Tok || !Tok->is(Next))
    return Range;
  return CharSourceRange::getTokenRange(Range.getBegin(), Tok->getLocation());
}
} // namespace internal

} // end namespace fixit
} // end namespace tooling
} // end namespace clang
