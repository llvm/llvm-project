//===--- SourceCode.cpp - Source code manipulation routines -----*- C++ -*-===//
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
#include "clang/Tooling/Transformer/SourceCode.h"
#include "clang/Lex/Lexer.h"

using namespace clang;

StringRef clang::tooling::getText(CharSourceRange Range,
                                  const ASTContext &Context) {
  return Lexer::getSourceText(Range, Context.getSourceManager(),
                              Context.getLangOpts());
}

CharSourceRange clang::tooling::maybeExtendRange(CharSourceRange Range,
                                                 tok::TokenKind Next,
                                                 ASTContext &Context) {
  Optional<Token> Tok = Lexer::findNextToken(
      Range.getEnd(), Context.getSourceManager(), Context.getLangOpts());
  if (!Tok || !Tok->is(Next))
    return Range;
  return CharSourceRange::getTokenRange(Range.getBegin(), Tok->getLocation());
}

llvm::Optional<CharSourceRange>
clang::tooling::getRangeForEdit(const CharSourceRange &EditRange,
                                const SourceManager &SM,
                                const LangOptions &LangOpts) {
  // FIXME: makeFileCharRange() has the disadvantage of stripping off "identity"
  // macros. For example, if we're looking to rewrite the int literal 3 to 6,
  // and we have the following definition:
  //    #define DO_NOTHING(x) x
  // then
  //    foo(DO_NOTHING(3))
  // will be rewritten to
  //    foo(6)
  // rather than the arguably better
  //    foo(DO_NOTHING(6))
  // Decide whether the current behavior is desirable and modify if not.
  CharSourceRange Range = Lexer::makeFileCharRange(EditRange, SM, LangOpts);
  if (Range.isInvalid())
    return None;

  if (Range.getBegin().isMacroID() || Range.getEnd().isMacroID())
    return None;
  if (SM.isInSystemHeader(Range.getBegin()) ||
      SM.isInSystemHeader(Range.getEnd()))
    return None;

  std::pair<FileID, unsigned> BeginInfo = SM.getDecomposedLoc(Range.getBegin());
  std::pair<FileID, unsigned> EndInfo = SM.getDecomposedLoc(Range.getEnd());
  if (BeginInfo.first != EndInfo.first ||
      BeginInfo.second > EndInfo.second)
    return None;

  return Range;
}
