//===--- IncludesSeparator.cpp ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements IncludesSeparator, a TokenAnalyzer that inserts
/// new lines or removes empty lines after an include area.
/// An includes area is a list of consecutive include statements.
///
//===----------------------------------------------------------------------===//

#include "IncludesSeparator.h"
#include "TokenAnnotator.h"
#define DEBUG_TYPE "includes-separator"

namespace {
bool isConditionalCompilationStart(const clang::format::AnnotatedLine &Line) {
  if (!Line.First)
    return false;
  const auto *NextToken = Line.First->getNextNonComment();
  return Line.First->is(clang::tok::hash) && NextToken &&
         NextToken->isOneOf(clang::tok::pp_if, clang::tok::pp_ifdef,
                            clang::tok::pp_ifndef, clang::tok::pp_defined);
}

bool isConditionalCompilationEnd(const clang::format::AnnotatedLine &Line) {
  if (!Line.First)
    return false;
  const auto *NextToken = Line.First->getNextNonComment();
  return Line.First->is(clang::tok::hash) && NextToken &&
         NextToken->is(clang::tok::pp_endif);
}

bool isConditionalCompilationStatement(
    const clang::format::AnnotatedLine &Line) {
  if (!Line.First)
    return false;
  const auto *NextToken = Line.First->getNextNonComment();
  return Line.First->is(clang::tok::hash) && NextToken &&
         NextToken->isOneOf(clang::tok::pp_if, clang::tok::pp_ifdef,
                            clang::tok::pp_ifndef, clang::tok::pp_elif,
                            clang::tok::pp_elifdef, clang::tok::pp_elifndef,
                            clang::tok::pp_else, clang::tok::pp_defined,
                            clang::tok::pp_endif);
}

bool isCCOnlyWithIncludes(
    const llvm::SmallVectorImpl<clang::format::AnnotatedLine *> &Lines,
    unsigned StartIdx) {
  int CCLevel = 0;
  for (unsigned I = StartIdx; I < Lines.size(); ++I) {
    const auto &CurrentLine = *Lines[I];
    if (isConditionalCompilationStart(CurrentLine))
      CCLevel++;

    if (isConditionalCompilationEnd(CurrentLine))
      CCLevel--;

    if (CCLevel == 0)
      break;

    if (!(CurrentLine.isInclude() ||
          isConditionalCompilationStatement(CurrentLine))) {
      return false;
    }
  }
  return true;
}

unsigned getEndOfCCBlock(
    const llvm::SmallVectorImpl<clang::format::AnnotatedLine *> &Lines,
    unsigned StartIdx) {
  int CCLevel = 0;
  unsigned I = StartIdx;
  for (; I < Lines.size(); ++I) {
    const auto &CurrentLine = *Lines[I];
    if (isConditionalCompilationStart(CurrentLine))
      CCLevel++;

    if (isConditionalCompilationEnd(CurrentLine))
      CCLevel--;

    if (CCLevel == 0)
      break;
  }
  return I;
}
} // namespace

namespace clang {
namespace format {
std::pair<tooling::Replacements, unsigned>
IncludesSeparator::analyze(TokenAnnotator &Annotator,
                           SmallVectorImpl<AnnotatedLine *> &AnnotatedLines,
                           FormatTokenLexer &Tokens) {
  assert(Style.EmptyLinesAfterIncludes.has_value());
  AffectedRangeMgr.computeAffectedLines(AnnotatedLines);
  tooling::Replacements Result;
  separateIncludes(AnnotatedLines, Result, Tokens);
  return {Result, 0};
}

void IncludesSeparator::separateIncludes(
    SmallVectorImpl<AnnotatedLine *> &Lines, tooling::Replacements &Result,
    FormatTokenLexer &Tokens) {
  const unsigned NewlineCount =
      std::min(Style.MaxEmptyLinesToKeep, *Style.EmptyLinesAfterIncludes) + 1;
  WhitespaceManager Whitespaces(
      Env.getSourceManager(), Style,
      Style.LineEnding > FormatStyle::LE_CRLF
          ? WhitespaceManager::inputUsesCRLF(
                Env.getSourceManager().getBufferData(Env.getFileID()),
                Style.LineEnding == FormatStyle::LE_DeriveCRLF)
          : Style.LineEnding == FormatStyle::LE_CRLF);

  bool InIncludeArea = false;
  for (unsigned I = 0; I < Lines.size(); ++I) {
    const auto &CurrentLine = *Lines[I];

    if (InIncludeArea) {
      if (CurrentLine.isInclude())
        continue;

      if (isConditionalCompilationStart(CurrentLine)) {
        const bool CCWithOnlyIncludes = isCCOnlyWithIncludes(Lines, I);
        I = getEndOfCCBlock(Lines, I);

        // Conditional compilation blocks that only contain
        // include statements are considered part of the include area.
        if (CCWithOnlyIncludes)
          continue;
      }

      if (!CurrentLine.First->is(tok::eof) && CurrentLine.Affected) {
        Whitespaces.replaceWhitespace(*CurrentLine.First, NewlineCount,
                                      CurrentLine.First->OriginalColumn,
                                      CurrentLine.First->OriginalColumn);
      }
      InIncludeArea = false;
    } else {
      if (CurrentLine.isInclude())
        InIncludeArea = true;
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
