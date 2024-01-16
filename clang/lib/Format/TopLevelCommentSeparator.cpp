//===--- TopLevelCommentSeparator.cpp ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements TopLevelCommentSeparator, a TokenAnalyzer that inserts
/// new lines or removes empty lines after the top level comment (i.e. comment
/// block at the top of the source file), usually license text or documentation.
///
//===----------------------------------------------------------------------===//

#include "TopLevelCommentSeparator.h"
#define DEBUG_TYPE "top-level-comment-separator"

namespace clang {
namespace format {
std::pair<tooling::Replacements, unsigned> TopLevelCommentSeparator::analyze(
    TokenAnnotator &Annotator, SmallVectorImpl<AnnotatedLine *> &AnnotatedLines,
    FormatTokenLexer &Tokens) {
  assert(Style.EmptyLinesAfterTopLevelComment.has_value());
  AffectedRangeMgr.computeAffectedLines(AnnotatedLines);
  tooling::Replacements Result;
  separateTopLevelComment(AnnotatedLines, Result, Tokens);
  return {Result, 0};
}

void TopLevelCommentSeparator::separateTopLevelComment(
    SmallVectorImpl<AnnotatedLine *> &Lines, tooling::Replacements &Result,
    FormatTokenLexer &Tokens) {
  unsigned NewlineCount = std::min(Style.MaxEmptyLinesToKeep,
                                   *Style.EmptyLinesAfterTopLevelComment) +
                          1;
  WhitespaceManager Whitespaces(
      Env.getSourceManager(), Style,
      Style.LineEnding > FormatStyle::LE_CRLF
          ? WhitespaceManager::inputUsesCRLF(
                Env.getSourceManager().getBufferData(Env.getFileID()),
                Style.LineEnding == FormatStyle::LE_DeriveCRLF)
          : Style.LineEnding == FormatStyle::LE_CRLF);

  bool InTopLevelComment = false;
  for (unsigned I = 0; I < Lines.size(); ++I) {
    const auto &CurrentLine = Lines[I];
    if (CurrentLine->isComment()) {
      InTopLevelComment = true;
    } else if (InTopLevelComment) {
      // Do not handle EOF newlines.
      if (!CurrentLine->First->is(tok::eof) && CurrentLine->Affected) {
        Whitespaces.replaceWhitespace(*CurrentLine->First, NewlineCount,
                                      CurrentLine->First->OriginalColumn,
                                      CurrentLine->First->OriginalColumn);
      }
      break;
    }
  }

  for (const auto &R : Whitespaces.generateReplacements()) {
    // The add method returns an Error instance which simulates program exit
    // code through overloading boolean operator, thus false here indicates
    // success.
    if (Result.add(R))
      return;
  }
}
} // namespace format
} // namespace clang
