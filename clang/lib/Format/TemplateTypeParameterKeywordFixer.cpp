//===--- TemplateTypeParameterKeywordFixer.cpp ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements TemplateTypeParameterKeywordFixer, a TokenAnalyzer
/// that rewrites \c typename and \c class when they introduce type template
/// parameters or template template parameters, according to
/// \c FormatStyle::TemplateTypeParameterKeyword.
///
//===----------------------------------------------------------------------===//

#include "TemplateTypeParameterKeywordFixer.h"
#include "FormatToken.h"
#include "clang/Format/Format.h"
#include "llvm/Support/Error.h"

using llvm::cantFail;

namespace clang {
namespace format {
namespace {

bool angleIntroducesTemplateParameterList(const FormatToken *LT) {
  const FormatToken *Prev = LT->getPreviousNonComment();
  if (!Prev)
    return false;
  if (Prev->is(tok::kw_template))
    return true;
  // Generic lambda
  return Prev->is(tok::r_square);
}

bool isStrictlyBetween(const SourceManager &SM, SourceLocation X,
                       SourceLocation Low, SourceLocation High) {
  return SM.isBeforeInTranslationUnit(Low, X) &&
         SM.isBeforeInTranslationUnit(X, High);
}

bool isInsideMatchingAngleRange(const FormatToken *Kw,
                                const SourceManager &SM) {
  SourceLocation KwLoc = Kw->getStartOfNonWhitespace();
  for (const FormatToken *T = Kw->getPreviousNonComment(); T;
       T = T->getPreviousNonComment()) {
    if (!T->is(tok::less))
      continue;
    if (!angleIntroducesTemplateParameterList(T))
      continue;
    if (!T->MatchingParen)
      continue;
    SourceLocation OpenLoc = T->getStartOfNonWhitespace();
    SourceLocation CloseLoc = T->MatchingParen->getStartOfNonWhitespace();
    if (isStrictlyBetween(SM, KwLoc, OpenLoc, CloseLoc))
      return true;
  }
  return false;
}

bool introducesTypeOrTemplateTemplateParameterName(const FormatToken *Kw) {
  const FormatToken *N = Kw->getNextNonComment();
  if (!N)
    return false;
  if (N->is(tok::ellipsis))
    N = N->getNextNonComment();
  if (!N)
    return false;
  if (N->isOneOf(tok::comma, tok::greater, tok::equal, tok::colon,
                  tok::kw_requires))
    return true;
  if (!N->Tok.getIdentifierInfo())
    return false;
  const FormatToken *AfterName = N->getNextNonComment();
  return !AfterName || !AfterName->is(tok::coloncolon);
}

bool prevIsTemplateParameterDelimiter(const FormatToken *Prev) {
  return Prev && Prev->isOneOf(tok::less, tok::comma, tok::greater);
}

/// \c true when \p Prev is the closing ``>`` of a nested ``template <...>``
/// prefix, so \p Kw introduces a template template parameter name (C++17
/// allows ``typename`` there; before C++17 only ``class`` is permitted).
bool isTemplateTemplateParameterIntroducer(const FormatToken *Prev) {
  return Prev && Prev->is(tok::greater);
}

bool allowsTypenameTemplateTemplateIntroducer(const FormatStyle &Style) {
  switch (Style.Standard) {
  case FormatStyle::LS_Auto:
  case FormatStyle::LS_Cpp11:
  case FormatStyle::LS_Cpp14:
    return false;
  default:
  return true;
  }
}

llvm::StringRef replacementKeyword(FormatStyle::TemplateTypeParameterKeywordOption O) {
  switch (O) {
  case FormatStyle::TTPS_UseTypename:
    return "typename";
  case FormatStyle::TTPS_UseClass:
    return "class";
  case FormatStyle::TTPS_Leave:
    break;
  }
  return {};
}

void processLine(AnnotatedLine *Line, const SourceManager &SM,
                 AffectedRangeManager &AffectedRangeMgr,
                 const FormatStyle &Style,
                 FormatStyle::TemplateTypeParameterKeywordOption Opt,
                 tooling::Replacements *Fixes) {
  if (!Line->Affected || Line->InPPDirective || Line->InMacroBody)
    return;

  for (FormatToken *Tok = Line->First; Tok; Tok = Tok->Next) {
    if (Tok->Finalized)
      continue;
    if (!Tok->isOneOf(tok::kw_typename, tok::kw_class))
      continue;

    const FormatToken *Prev = Tok->getPreviousNonComment();
    if (!prevIsTemplateParameterDelimiter(Prev))
      continue;
    if (!isInsideMatchingAngleRange(Tok, SM))
      continue;
    if (!introducesTypeOrTemplateTemplateParameterName(Tok))
      continue;
    if (isTemplateTemplateParameterIntroducer(Prev) && !allowsTypenameTemplateTemplateIntroducer(Style))
      continue;

    llvm::StringRef NewText = replacementKeyword(Opt);
    if (NewText.empty() || NewText == Tok->TokenText)
      continue;

    SourceLocation Loc = Tok->Tok.getLocation();
    unsigned Length = Tok->TokenText.size();
    if (!AffectedRangeMgr.affectsCharSourceRange(
            CharSourceRange::getCharRange(Loc, Loc.getLocWithOffset(Length))))
      continue;

    cantFail(Fixes->add(tooling::Replacement(SM, Loc, Length, NewText.str())));
  }

  for (AnnotatedLine *Child : Line->Children)
    processLine(Child, SM, AffectedRangeMgr, Style, Opt, Fixes);
}

} // namespace

TemplateTypeParameterKeywordFixer::TemplateTypeParameterKeywordFixer(
    const Environment &Env, const FormatStyle &Style)
    : TokenAnalyzer(Env, Style) {}

std::pair<tooling::Replacements, unsigned>
TemplateTypeParameterKeywordFixer::analyze(
    TokenAnnotator &, SmallVectorImpl<AnnotatedLine *> &AnnotatedLines,
    FormatTokenLexer &) {
  if (Style.TemplateTypeParameterKeyword == FormatStyle::TTPS_Leave)
    return {};

  AffectedRangeMgr.computeAffectedLines(AnnotatedLines);
  tooling::Replacements Fixes;
  const SourceManager &SM = Env.getSourceManager();

  for (AnnotatedLine *Line : AnnotatedLines)
    processLine(Line, SM, AffectedRangeMgr, Style,
                Style.TemplateTypeParameterKeyword, &Fixes);

  return {Fixes, 0};
}

} // namespace format
} // namespace clang
