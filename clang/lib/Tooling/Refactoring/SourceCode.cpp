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
#include "clang/Tooling/Refactoring/SourceCode.h"
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
