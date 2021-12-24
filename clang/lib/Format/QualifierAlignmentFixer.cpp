//===--- LeftRightQualifierAlignmentFixer.cpp -------------------*- C++--*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements LeftRightQualifierAlignmentFixer, a TokenAnalyzer that
/// enforces either left or right const depending on the style.
///
//===----------------------------------------------------------------------===//

#include "QualifierAlignmentFixer.h"
#include "FormatToken.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Regex.h"

#include <algorithm>

#define DEBUG_TYPE "format-qualifier-alignment-fixer"

namespace clang {
namespace format {

QualifierAlignmentFixer::QualifierAlignmentFixer(
    const Environment &Env, const FormatStyle &Style, StringRef &Code,
    ArrayRef<tooling::Range> Ranges, unsigned FirstStartColumn,
    unsigned NextStartColumn, unsigned LastStartColumn, StringRef FileName)
    : TokenAnalyzer(Env, Style), Code(Code), Ranges(Ranges),
      FirstStartColumn(FirstStartColumn), NextStartColumn(NextStartColumn),
      LastStartColumn(LastStartColumn), FileName(FileName) {
  std::vector<std::string> LeftOrder;
  std::vector<std::string> RightOrder;
  std::vector<tok::TokenKind> ConfiguredQualifierTokens;
  PrepareLeftRightOrdering(Style.QualifierOrder, LeftOrder, RightOrder,
                           ConfiguredQualifierTokens);

  // Handle the left and right Alignment Seperately
  for (const auto &Qualifier : LeftOrder) {
    Passes.emplace_back(
        [&, Qualifier, ConfiguredQualifierTokens](const Environment &Env) {
          return LeftRightQualifierAlignmentFixer(Env, Style, Qualifier,
                                                  ConfiguredQualifierTokens,
                                                  /*RightAlign=*/false)
              .process();
        });
  }
  for (const auto &Qualifier : RightOrder) {
    Passes.emplace_back(
        [&, Qualifier, ConfiguredQualifierTokens](const Environment &Env) {
          return LeftRightQualifierAlignmentFixer(Env, Style, Qualifier,
                                                  ConfiguredQualifierTokens,
                                                  /*RightAlign=*/true)
              .process();
        });
  }
}

std::pair<tooling::Replacements, unsigned> QualifierAlignmentFixer::analyze(
    TokenAnnotator &Annotator, SmallVectorImpl<AnnotatedLine *> &AnnotatedLines,
    FormatTokenLexer &Tokens) {
  auto Env = Environment::make(Code, FileName, Ranges, FirstStartColumn,
                               NextStartColumn, LastStartColumn);
  if (!Env)
    return {};
  llvm::Optional<std::string> CurrentCode = None;
  tooling::Replacements Fixes;
  for (size_t I = 0, E = Passes.size(); I < E; ++I) {
    std::pair<tooling::Replacements, unsigned> PassFixes = Passes[I](*Env);
    auto NewCode = applyAllReplacements(
        CurrentCode ? StringRef(*CurrentCode) : Code, PassFixes.first);
    if (NewCode) {
      Fixes = Fixes.merge(PassFixes.first);
      if (I + 1 < E) {
        CurrentCode = std::move(*NewCode);
        Env = Environment::make(
            *CurrentCode, FileName,
            tooling::calculateRangesAfterReplacements(Fixes, Ranges),
            FirstStartColumn, NextStartColumn, LastStartColumn);
        if (!Env)
          return {};
      }
    }
  }

  // Don't make replacements that replace nothing.
  tooling::Replacements NonNoOpFixes;

  for (auto I = Fixes.begin(), E = Fixes.end(); I != E; ++I) {
    StringRef OriginalCode = Code.substr(I->getOffset(), I->getLength());

    if (!OriginalCode.equals(I->getReplacementText())) {
      auto Err = NonNoOpFixes.add(*I);
      if (Err)
        llvm::errs() << "Error adding replacements : "
                     << llvm::toString(std::move(Err)) << "\n";
    }
  }
  return {NonNoOpFixes, 0};
}

static void replaceToken(const SourceManager &SourceMgr,
                         tooling::Replacements &Fixes,
                         const CharSourceRange &Range, std::string NewText) {
  auto Replacement = tooling::Replacement(SourceMgr, Range, NewText);
  auto Err = Fixes.add(Replacement);

  if (Err)
    llvm::errs() << "Error while rearranging Qualifier : "
                 << llvm::toString(std::move(Err)) << "\n";
}

static void removeToken(const SourceManager &SourceMgr,
                        tooling::Replacements &Fixes,
                        const FormatToken *First) {
  auto Range = CharSourceRange::getCharRange(First->getStartOfNonWhitespace(),
                                             First->Tok.getEndLoc());
  replaceToken(SourceMgr, Fixes, Range, "");
}

static void insertQualifierAfter(const SourceManager &SourceMgr,
                                 tooling::Replacements &Fixes,
                                 const FormatToken *First,
                                 const std::string &Qualifier) {
  FormatToken *Next = First->Next;
  if (!Next)
    return;
  auto Range = CharSourceRange::getCharRange(Next->getStartOfNonWhitespace(),
                                             Next->Tok.getEndLoc());

  std::string NewText = " " + Qualifier + " ";
  NewText += Next->TokenText;
  replaceToken(SourceMgr, Fixes, Range, NewText);
}

static void insertQualifierBefore(const SourceManager &SourceMgr,
                                  tooling::Replacements &Fixes,
                                  const FormatToken *First,
                                  const std::string &Qualifier) {
  auto Range = CharSourceRange::getCharRange(First->getStartOfNonWhitespace(),
                                             First->Tok.getEndLoc());

  std::string NewText = " " + Qualifier + " ";
  NewText += First->TokenText;

  replaceToken(SourceMgr, Fixes, Range, NewText);
}

static bool endsWithSpace(const std::string &s) {
  if (s.empty()) {
    return false;
  }
  return isspace(s.back());
}

static bool startsWithSpace(const std::string &s) {
  if (s.empty()) {
    return false;
  }
  return isspace(s.front());
}

static void rotateTokens(const SourceManager &SourceMgr,
                         tooling::Replacements &Fixes, const FormatToken *First,
                         const FormatToken *Last, bool Left) {
  auto *End = Last;
  auto *Begin = First;
  if (!Left) {
    End = Last->Next;
    Begin = First->Next;
  }

  std::string NewText;
  // If we are rotating to the left we move the Last token to the front.
  if (Left) {
    NewText += Last->TokenText;
    NewText += " ";
  }

  // Then move through the other tokens.
  auto *Tok = Begin;
  while (Tok != End) {
    if (!NewText.empty() && !endsWithSpace(NewText)) {
      NewText += " ";
    }

    NewText += Tok->TokenText;
    Tok = Tok->Next;
  }

  // If we are rotating to the right we move the first token to the back.
  if (!Left) {
    if (!NewText.empty() && !startsWithSpace(NewText)) {
      NewText += " ";
    }
    NewText += First->TokenText;
  }

  auto Range = CharSourceRange::getCharRange(First->getStartOfNonWhitespace(),
                                             Last->Tok.getEndLoc());

  replaceToken(SourceMgr, Fixes, Range, NewText);
}

FormatToken *LeftRightQualifierAlignmentFixer::analyzeRight(
    const SourceManager &SourceMgr, const AdditionalKeywords &Keywords,
    tooling::Replacements &Fixes, FormatToken *Tok,
    const std::string &Qualifier, tok::TokenKind QualifierType) {
  // We only need to think about streams that begin with a qualifier.
  if (!Tok->is(QualifierType))
    return Tok;
  // Don't concern yourself if nothing follows the qualifier.
  if (!Tok->Next)
    return Tok;
  if (LeftRightQualifierAlignmentFixer::isPossibleMacro(Tok->Next))
    return Tok;

  FormatToken *Qual = Tok->Next;
  FormatToken *LastQual = Qual;
  while (Qual && isQualifierOrType(Qual, ConfiguredQualifierTokens)) {
    LastQual = Qual;
    Qual = Qual->Next;
  }
  if (LastQual && Qual != LastQual) {
    rotateTokens(SourceMgr, Fixes, Tok, LastQual, /*Left=*/false);
    Tok = LastQual;
  } else if (Tok->startsSequence(QualifierType, tok::identifier,
                                 TT_TemplateOpener)) {
    // Read from the TemplateOpener to
    // TemplateCloser as in const ArrayRef<int> a; const ArrayRef<int> &a;
    FormatToken *EndTemplate = Tok->Next->Next->MatchingParen;
    if (EndTemplate) {
      // Move to the end of any template class members e.g.
      // `Foo<int>::iterator`.
      if (EndTemplate->startsSequence(TT_TemplateCloser, tok::coloncolon,
                                      tok::identifier))
        EndTemplate = EndTemplate->Next->Next;
    }
    if (EndTemplate && EndTemplate->Next &&
        !EndTemplate->Next->isOneOf(tok::equal, tok::l_paren)) {
      insertQualifierAfter(SourceMgr, Fixes, EndTemplate, Qualifier);
      // Remove the qualifier.
      removeToken(SourceMgr, Fixes, Tok);
      return Tok;
    }
  } else if (Tok->startsSequence(QualifierType, tok::identifier)) {
    FormatToken *Next = Tok->Next;
    // The case  `const Foo` -> `Foo const`
    // The case  `const Foo *` -> `Foo const *`
    // The case  `const Foo &` -> `Foo const &`
    // The case  `const Foo &&` -> `Foo const &&`
    // The case  `const std::Foo &&` -> `std::Foo const &&`
    // The case  `const std::Foo<T> &&` -> `std::Foo<T> const &&`
    while (Next && Next->isOneOf(tok::identifier, tok::coloncolon)) {
      Next = Next->Next;
    }
    if (Next && Next->is(TT_TemplateOpener)) {
      Next = Next->MatchingParen;
      // Move to the end of any template class members e.g.
      // `Foo<int>::iterator`.
      if (Next && Next->startsSequence(TT_TemplateCloser, tok::coloncolon,
                                       tok::identifier)) {
        Next = Next->Next->Next;
        return Tok;
      }
      assert(Next && "Missing template opener");
      Next = Next->Next;
    }
    if (Next && Next->isOneOf(tok::star, tok::amp, tok::ampamp) &&
        !Tok->Next->isOneOf(Keywords.kw_override, Keywords.kw_final)) {
      if (Next->Previous && !Next->Previous->is(QualifierType)) {
        insertQualifierAfter(SourceMgr, Fixes, Next->Previous, Qualifier);
        removeToken(SourceMgr, Fixes, Tok);
      }
      return Next;
    }
  }

  return Tok;
}

FormatToken *LeftRightQualifierAlignmentFixer::analyzeLeft(
    const SourceManager &SourceMgr, const AdditionalKeywords &Keywords,
    tooling::Replacements &Fixes, FormatToken *Tok,
    const std::string &Qualifier, tok::TokenKind QualifierType) {
  // if Tok is an identifier and possibly a macro then don't convert.
  if (LeftRightQualifierAlignmentFixer::isPossibleMacro(Tok))
    return Tok;

  FormatToken *Qual = Tok;
  FormatToken *LastQual = Qual;
  while (Qual && isQualifierOrType(Qual, ConfiguredQualifierTokens)) {
    LastQual = Qual;
    Qual = Qual->Next;
    if (Qual && Qual->is(QualifierType))
      break;
  }

  if (!Qual) {
    return Tok;
  }

  if (LastQual && Qual != LastQual && Qual->is(QualifierType)) {
    rotateTokens(SourceMgr, Fixes, Tok, Qual, /*Left=*/true);
    Tok = Qual->Next;
  } else if (Tok->startsSequence(tok::identifier, QualifierType)) {
    if (Tok->Next->Next && Tok->Next->Next->isOneOf(tok::identifier, tok::star,
                                                    tok::amp, tok::ampamp)) {
      // Don't swap `::iterator const` to `::const iterator`.
      if (!Tok->Previous ||
          (Tok->Previous && !Tok->Previous->is(tok::coloncolon))) {
        rotateTokens(SourceMgr, Fixes, Tok, Tok->Next, /*Left=*/true);
        Tok = Tok->Next;
      }
    }
  }
  if (Tok->is(TT_TemplateOpener) && Tok->Next &&
      (Tok->Next->is(tok::identifier) || Tok->Next->isSimpleTypeSpecifier()) &&
      Tok->Next->Next && Tok->Next->Next->is(QualifierType)) {
    rotateTokens(SourceMgr, Fixes, Tok->Next, Tok->Next->Next, /*Left=*/true);
  }
  if (Tok->startsSequence(tok::identifier) && Tok->Next) {
    if (Tok->Previous &&
        Tok->Previous->isOneOf(tok::star, tok::ampamp, tok::amp)) {
      return Tok;
    }
    FormatToken *Next = Tok->Next;
    // The case  `std::Foo<T> const` -> `const std::Foo<T> &&`
    while (Next && Next->isOneOf(tok::identifier, tok::coloncolon))
      Next = Next->Next;
    if (Next && Next->Previous &&
        Next->Previous->startsSequence(tok::identifier, TT_TemplateOpener)) {
      // Read from to the end of the TemplateOpener to
      // TemplateCloser const ArrayRef<int> a; const ArrayRef<int> &a;
      assert(Next->MatchingParen && "Missing template closer");
      Next = Next->MatchingParen->Next;

      // Move to the end of any template class members e.g.
      // `Foo<int>::iterator`.
      if (Next && Next->startsSequence(tok::coloncolon, tok::identifier))
        Next = Next->Next->Next;
      if (Next && Next->is(QualifierType)) {
        // Remove the const.
        insertQualifierBefore(SourceMgr, Fixes, Tok, Qualifier);
        removeToken(SourceMgr, Fixes, Next);
        return Next;
      }
    }
    if (Next && Next->Next &&
        Next->Next->isOneOf(tok::amp, tok::ampamp, tok::star)) {
      if (Next->is(QualifierType)) {
        // Remove the qualifier.
        insertQualifierBefore(SourceMgr, Fixes, Tok, Qualifier);
        removeToken(SourceMgr, Fixes, Next);
        return Next;
      }
    }
  }
  return Tok;
}

tok::TokenKind LeftRightQualifierAlignmentFixer::getTokenFromQualifier(
    const std::string &Qualifier) {
  // Don't let 'type' be an identifier, but steal typeof token.
  return llvm::StringSwitch<tok::TokenKind>(Qualifier)
      .Case("type", tok::kw_typeof)
      .Case("const", tok::kw_const)
      .Case("volatile", tok::kw_volatile)
      .Case("static", tok::kw_static)
      .Case("inline", tok::kw_inline)
      .Case("constexpr", tok::kw_constexpr)
      .Case("restrict", tok::kw_restrict)
      .Default(tok::identifier);
}

LeftRightQualifierAlignmentFixer::LeftRightQualifierAlignmentFixer(
    const Environment &Env, const FormatStyle &Style,
    const std::string &Qualifier,
    const std::vector<tok::TokenKind> &QualifierTokens, bool RightAlign)
    : TokenAnalyzer(Env, Style), Qualifier(Qualifier), RightAlign(RightAlign),
      ConfiguredQualifierTokens(QualifierTokens) {}

std::pair<tooling::Replacements, unsigned>
LeftRightQualifierAlignmentFixer::analyze(
    TokenAnnotator &Annotator, SmallVectorImpl<AnnotatedLine *> &AnnotatedLines,
    FormatTokenLexer &Tokens) {
  tooling::Replacements Fixes;
  const AdditionalKeywords &Keywords = Tokens.getKeywords();
  const SourceManager &SourceMgr = Env.getSourceManager();
  AffectedRangeMgr.computeAffectedLines(AnnotatedLines);

  tok::TokenKind QualifierToken = getTokenFromQualifier(Qualifier);
  assert(QualifierToken != tok::identifier && "Unrecognised Qualifier");

  for (size_t I = 0, E = AnnotatedLines.size(); I != E; ++I) {
    FormatToken *First = AnnotatedLines[I]->First;
    const auto *Last = AnnotatedLines[I]->Last;

    for (auto *Tok = First; Tok && Tok != Last && Tok->Next; Tok = Tok->Next) {
      if (Tok->is(tok::comment))
        continue;
      if (RightAlign)
        Tok = analyzeRight(SourceMgr, Keywords, Fixes, Tok, Qualifier,
                           QualifierToken);
      else
        Tok = analyzeLeft(SourceMgr, Keywords, Fixes, Tok, Qualifier,
                          QualifierToken);
    }
  }
  return {Fixes, 0};
}

void QualifierAlignmentFixer::PrepareLeftRightOrdering(
    const std::vector<std::string> &Order, std::vector<std::string> &LeftOrder,
    std::vector<std::string> &RightOrder,
    std::vector<tok::TokenKind> &Qualifiers) {

  // Depending on the position of type in the order you need
  // To iterate forward or backward through the order list as qualifier
  // can push through each other.
  // The Order list must define the position of "type" to signify
  assert(llvm::is_contained(Order, "type") &&
         "QualifierOrder must contain type");
  // Split the Order list by type and reverse the left side.

  bool left = true;
  for (const auto &s : Order) {
    if (s == "type") {
      left = false;
      continue;
    }

    tok::TokenKind QualifierToken =
        LeftRightQualifierAlignmentFixer::getTokenFromQualifier(s);
    if (QualifierToken != tok::kw_typeof && QualifierToken != tok::identifier) {
      Qualifiers.push_back(QualifierToken);
    }

    if (left)
      // Reverse the order for left aligned items.
      LeftOrder.insert(LeftOrder.begin(), s);
    else
      RightOrder.push_back(s);
  }
}

bool LeftRightQualifierAlignmentFixer::isQualifierOrType(
    const FormatToken *Tok, const std::vector<tok::TokenKind> &specifiedTypes) {
  return Tok && (Tok->isSimpleTypeSpecifier() || Tok->is(tok::kw_auto) ||
                 llvm::is_contained(specifiedTypes, Tok->Tok.getKind()));
}

// If a token is an identifier and it's upper case, it could
// be a macro and hence we need to be able to ignore it.
bool LeftRightQualifierAlignmentFixer::isPossibleMacro(const FormatToken *Tok) {
  if (!Tok)
    return false;
  if (!Tok->is(tok::identifier))
    return false;
  if (Tok->TokenText.upper() == Tok->TokenText.str())
    return true;
  return false;
}

} // namespace format
} // namespace clang
