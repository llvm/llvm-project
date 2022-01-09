//===--- DefinitionBlockSeparator.cpp ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements DefinitionBlockSeparator, a TokenAnalyzer that inserts
/// or removes empty lines separating definition blocks like classes, structs,
/// functions, enums, and namespaces in between.
///
//===----------------------------------------------------------------------===//

#include "DefinitionBlockSeparator.h"
#include "llvm/Support/Debug.h"
#define DEBUG_TYPE "definition-block-separator"

namespace clang {
namespace format {
std::pair<tooling::Replacements, unsigned> DefinitionBlockSeparator::analyze(
    TokenAnnotator &Annotator, SmallVectorImpl<AnnotatedLine *> &AnnotatedLines,
    FormatTokenLexer &Tokens) {
  assert(Style.SeparateDefinitionBlocks != FormatStyle::SDS_Leave);
  AffectedRangeMgr.computeAffectedLines(AnnotatedLines);
  tooling::Replacements Result;
  separateBlocks(AnnotatedLines, Result);
  return {Result, 0};
}

void DefinitionBlockSeparator::separateBlocks(
    SmallVectorImpl<AnnotatedLine *> &Lines, tooling::Replacements &Result) {
  auto LikelyDefinition = [this](const AnnotatedLine *Line) {
    if (Line->MightBeFunctionDecl && Line->mightBeFunctionDefinition())
      return true;
    FormatToken *CurrentToken = Line->First;
    while (CurrentToken) {
      if (CurrentToken->isOneOf(tok::kw_class, tok::kw_struct,
                                tok::kw_namespace, tok::kw_enum) ||
          (Style.Language == FormatStyle::LK_JavaScript &&
           CurrentToken->TokenText == "function"))
        return true;
      CurrentToken = CurrentToken->Next;
    }
    return false;
  };
  unsigned NewlineCount =
      (Style.SeparateDefinitionBlocks == FormatStyle::SDS_Always ? 1 : 0) + 1;
  WhitespaceManager Whitespaces(
      Env.getSourceManager(), Style,
      Style.DeriveLineEnding
          ? WhitespaceManager::inputUsesCRLF(
                Env.getSourceManager().getBufferData(Env.getFileID()),
                Style.UseCRLF)
          : Style.UseCRLF);
  for (unsigned I = 0; I < Lines.size(); I++) {
    const auto &CurrentLine = Lines[I];
    FormatToken *TargetToken = nullptr;
    AnnotatedLine *TargetLine;
    auto OpeningLineIndex = CurrentLine->MatchingOpeningBlockLineIndex;
    const auto InsertReplacement = [&](const int NewlineToInsert) {
      assert(TargetLine);
      assert(TargetToken);

      // Do not handle EOF newlines.
      if (TargetToken->is(tok::eof) && NewlineToInsert > 0)
        return;
      if (!TargetLine->Affected)
        return;
      Whitespaces.replaceWhitespace(*TargetToken, NewlineToInsert,
                                    TargetToken->SpacesRequiredBefore - 1,
                                    TargetToken->StartsColumn);
    };
    const auto FollowingOtherOpening = [&]() {
      return OpeningLineIndex == 0 ||
             Lines[OpeningLineIndex - 1]->Last->opensScope();
    };
    const auto HasEnumOnLine = [CurrentLine]() {
      FormatToken *CurrentToken = CurrentLine->First;
      while (CurrentToken) {
        if (CurrentToken->is(tok::kw_enum))
          return true;
        CurrentToken = CurrentToken->Next;
      }
      return false;
    };

    bool IsDefBlock = false;

    if (HasEnumOnLine()) {
      // We have no scope opening/closing information for enum.
      IsDefBlock = true;
      OpeningLineIndex = I;
      TargetLine = CurrentLine;
      TargetToken = CurrentLine->First;
      if (!FollowingOtherOpening())
        InsertReplacement(NewlineCount);
      else
        InsertReplacement(OpeningLineIndex != 0);
      while (TargetToken && !TargetToken->is(tok::r_brace))
        TargetToken = TargetToken->Next;
      if (!TargetToken) {
        while (I < Lines.size() && !Lines[I]->First->is(tok::r_brace))
          ++I;
      }
    } else if (CurrentLine->First->closesScope()) {
      if (OpeningLineIndex > Lines.size())
        continue;
      // Handling the case that opening bracket has its own line.
      OpeningLineIndex -= Lines[OpeningLineIndex]->First->TokenText == "{";
      AnnotatedLine *OpeningLine = Lines[OpeningLineIndex];
      // Closing a function definition.
      if (LikelyDefinition(OpeningLine)) {
        IsDefBlock = true;
        if (OpeningLineIndex > 0) {
          OpeningLineIndex -=
              Style.Language == FormatStyle::LK_CSharp &&
              Lines[OpeningLineIndex - 1]->First->is(tok::l_square);
          OpeningLine = Lines[OpeningLineIndex];
        }
        TargetLine = OpeningLine;
        TargetToken = TargetLine->First;
        if (!FollowingOtherOpening()) {
          // Avoid duplicated replacement.
          if (!TargetToken->opensScope())
            InsertReplacement(NewlineCount);
        } else
          InsertReplacement(OpeningLineIndex != 0);
      }
    }

    // Not the last token.
    if (IsDefBlock && I + 1 < Lines.size()) {
      TargetLine = Lines[I + 1];
      TargetToken = TargetLine->First;

      // No empty line for continuously closing scopes. The token will be
      // handled in another case if the line following is opening a
      // definition.
      if (!TargetToken->closesScope()) {
        if (!LikelyDefinition(TargetLine))
          InsertReplacement(NewlineCount);
      } else {
        InsertReplacement(OpeningLineIndex != 0);
      }
    }
  }
  for (const auto &R : Whitespaces.generateReplacements())
    // The add method returns an Error instance which simulates program exit
    // code through overloading boolean operator, thus false here indicates
    // success.
    if (Result.add(R))
      return;
}
} // namespace format
} // namespace clang
