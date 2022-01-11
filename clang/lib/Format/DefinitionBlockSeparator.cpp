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
  const bool IsNeverStyle =
      Style.SeparateDefinitionBlocks == FormatStyle::SDS_Never;
  auto LikelyDefinition = [this](const AnnotatedLine *Line) {
    if ((Line->MightBeFunctionDecl && Line->mightBeFunctionDefinition()) ||
        Line->startsWithNamespace())
      return true;
    FormatToken *CurrentToken = Line->First;
    while (CurrentToken) {
      if (CurrentToken->isOneOf(tok::kw_class, tok::kw_struct, tok::kw_enum) ||
          (Style.isJavaScript() && CurrentToken->TokenText == "function"))
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
  for (unsigned I = 0; I < Lines.size(); ++I) {
    const auto &CurrentLine = Lines[I];
    if (CurrentLine->InPPDirective)
      continue;
    FormatToken *TargetToken = nullptr;
    AnnotatedLine *TargetLine;
    auto OpeningLineIndex = CurrentLine->MatchingOpeningBlockLineIndex;
    AnnotatedLine *OpeningLine = nullptr;
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
    const auto IsPPConditional = [&](const size_t LineIndex) {
      const auto &Line = Lines[LineIndex];
      return Line->First->is(tok::hash) && Line->First->Next &&
             Line->First->Next->isOneOf(tok::pp_if, tok::pp_ifdef, tok::pp_else,
                                        tok::pp_ifndef, tok::pp_elifndef,
                                        tok::pp_elifdef, tok::pp_elif,
                                        tok::pp_endif);
    };
    const auto FollowingOtherOpening = [&]() {
      return OpeningLineIndex == 0 ||
             Lines[OpeningLineIndex - 1]->Last->opensScope() ||
             IsPPConditional(OpeningLineIndex - 1);
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
    const auto MayPrecedeDefinition = [&](const int Direction = -1) {
      const size_t OperateIndex = OpeningLineIndex + Direction;
      assert(OperateIndex < Lines.size());
      const auto &OperateLine = Lines[OperateIndex];
      return (Style.isCSharp() && OperateLine->First->is(TT_AttributeSquare)) ||
             OperateLine->First->is(tok::comment);
    };

    if (HasEnumOnLine()) {
      // We have no scope opening/closing information for enum.
      IsDefBlock = true;
      OpeningLineIndex = I;
      while (OpeningLineIndex > 0 && MayPrecedeDefinition())
        --OpeningLineIndex;
      OpeningLine = Lines[OpeningLineIndex];
      TargetLine = OpeningLine;
      TargetToken = TargetLine->First;
      if (!FollowingOtherOpening())
        InsertReplacement(NewlineCount);
      else if (IsNeverStyle)
        InsertReplacement(OpeningLineIndex != 0);
      TargetLine = CurrentLine;
      TargetToken = TargetLine->First;
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
      OpeningLineIndex -= Lines[OpeningLineIndex]->First->is(tok::l_brace);
      OpeningLine = Lines[OpeningLineIndex];
      // Closing a function definition.
      if (LikelyDefinition(OpeningLine)) {
        IsDefBlock = true;
        while (OpeningLineIndex > 0 && MayPrecedeDefinition())
          --OpeningLineIndex;
        OpeningLine = Lines[OpeningLineIndex];
        TargetLine = OpeningLine;
        TargetToken = TargetLine->First;
        if (!FollowingOtherOpening()) {
          // Avoid duplicated replacement.
          if (TargetToken->isNot(tok::l_brace))
            InsertReplacement(NewlineCount);
        } else if (IsNeverStyle)
          InsertReplacement(OpeningLineIndex != 0);
      }
    }

    // Not the last token.
    if (IsDefBlock && I + 1 < Lines.size()) {
      OpeningLineIndex = I + 1;
      TargetLine = Lines[OpeningLineIndex];
      TargetToken = TargetLine->First;

      // No empty line for continuously closing scopes. The token will be
      // handled in another case if the line following is opening a
      // definition.
      if (!TargetToken->closesScope() && !IsPPConditional(OpeningLineIndex)) {
        // Check whether current line may be precedings of a definition line.
        while (OpeningLineIndex + 1 < Lines.size() &&
               MayPrecedeDefinition(/*Direction=*/0))
          ++OpeningLineIndex;
        TargetLine = Lines[OpeningLineIndex];
        if (!LikelyDefinition(TargetLine)) {
          TargetLine = Lines[I + 1];
          TargetToken = TargetLine->First;
          InsertReplacement(NewlineCount);
        }
      } else if (IsNeverStyle) {
        TargetLine = Lines[I + 1];
        TargetToken = TargetLine->First;
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
