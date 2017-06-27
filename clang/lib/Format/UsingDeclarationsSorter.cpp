//===--- UsingDeclarationsSorter.cpp ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file implements UsingDeclarationsSorter, a TokenAnalyzer that
/// sorts consecutive using declarations.
///
//===----------------------------------------------------------------------===//

#include "UsingDeclarationsSorter.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Regex.h"

#include <algorithm>

#define DEBUG_TYPE "using-declarations-sorter"

namespace clang {
namespace format {

namespace {

struct UsingDeclaration {
  const AnnotatedLine *Line;
  std::string Label;

  UsingDeclaration(const AnnotatedLine *Line, const std::string &Label)
      : Line(Line), Label(Label) {}

  bool operator<(const UsingDeclaration &Other) const {
    return Label < Other.Label;
  }
};

/// Computes the label of a using declaration starting at tthe using token
/// \p UsingTok.
/// If \p UsingTok doesn't begin a using declaration, returns the empty string.
/// Note that this detects specifically using declarations, as in:
/// using A::B::C;
/// and not type aliases, as in:
/// using A = B::C;
/// Type aliases are in general not safe to permute.
std::string computeUsingDeclarationLabel(const FormatToken *UsingTok) {
  assert(UsingTok && UsingTok->is(tok::kw_using) && "Expecting a using token");
  std::string Label;
  const FormatToken *Tok = UsingTok->Next;
  if (Tok && Tok->is(tok::kw_typename)) {
    Label.append("typename ");
    Tok = Tok->Next;
  }
  if (Tok && Tok->is(tok::coloncolon)) {
    Label.append("::");
    Tok = Tok->Next;
  }
  bool HasIdentifier = false;
  while (Tok && Tok->is(tok::identifier)) {
    HasIdentifier = true;
    Label.append(Tok->TokenText.str());
    Tok = Tok->Next;
    if (!Tok || Tok->isNot(tok::coloncolon))
      break;
    Label.append("::");
    Tok = Tok->Next;
  }
  if (HasIdentifier && Tok && Tok->isOneOf(tok::semi, tok::comma))
    return Label;
  return "";
}

void endUsingDeclarationBlock(
    SmallVectorImpl<UsingDeclaration> *UsingDeclarations,
    const SourceManager &SourceMgr, tooling::Replacements *Fixes) {
  SmallVector<UsingDeclaration, 4> SortedUsingDeclarations(
      UsingDeclarations->begin(), UsingDeclarations->end());
  std::sort(SortedUsingDeclarations.begin(), SortedUsingDeclarations.end());
  for (size_t I = 0, E = UsingDeclarations->size(); I < E; ++I) {
    if ((*UsingDeclarations)[I].Line == SortedUsingDeclarations[I].Line)
      continue;
    auto Begin = (*UsingDeclarations)[I].Line->First->Tok.getLocation();
    auto End = (*UsingDeclarations)[I].Line->Last->Tok.getEndLoc();
    auto SortedBegin =
        SortedUsingDeclarations[I].Line->First->Tok.getLocation();
    auto SortedEnd = SortedUsingDeclarations[I].Line->Last->Tok.getEndLoc();
    StringRef Text(SourceMgr.getCharacterData(SortedBegin),
                   SourceMgr.getCharacterData(SortedEnd) -
                       SourceMgr.getCharacterData(SortedBegin));
    DEBUG({
      StringRef OldText(SourceMgr.getCharacterData(Begin),
                        SourceMgr.getCharacterData(End) -
                            SourceMgr.getCharacterData(Begin));
      llvm::dbgs() << "Replacing '" << OldText << "' with '" << Text << "'\n";
    });
    auto Range = CharSourceRange::getCharRange(Begin, End);
    auto Err = Fixes->add(tooling::Replacement(SourceMgr, Range, Text));
    if (Err) {
      llvm::errs() << "Error while sorting using declarations: "
                   << llvm::toString(std::move(Err)) << "\n";
    }
  }
  UsingDeclarations->clear();
}

} // namespace

UsingDeclarationsSorter::UsingDeclarationsSorter(const Environment &Env,
                                                 const FormatStyle &Style)
    : TokenAnalyzer(Env, Style) {}

tooling::Replacements UsingDeclarationsSorter::analyze(
    TokenAnnotator &Annotator, SmallVectorImpl<AnnotatedLine *> &AnnotatedLines,
    FormatTokenLexer &Tokens) {
  const SourceManager &SourceMgr = Env.getSourceManager();
  AffectedRangeMgr.computeAffectedLines(AnnotatedLines.begin(),
                                        AnnotatedLines.end());
  tooling::Replacements Fixes;
  SmallVector<UsingDeclaration, 4> UsingDeclarations;
  for (size_t I = 0, E = AnnotatedLines.size(); I != E; ++I) {
    if (!AnnotatedLines[I]->Affected || AnnotatedLines[I]->InPPDirective ||
        !AnnotatedLines[I]->startsWith(tok::kw_using) ||
        AnnotatedLines[I]->First->Finalized) {
      endUsingDeclarationBlock(&UsingDeclarations, SourceMgr, &Fixes);
      continue;
    }
    if (AnnotatedLines[I]->First->NewlinesBefore > 1)
      endUsingDeclarationBlock(&UsingDeclarations, SourceMgr, &Fixes);
    std::string Label = computeUsingDeclarationLabel(AnnotatedLines[I]->First);
    if (Label.empty()) {
      endUsingDeclarationBlock(&UsingDeclarations, SourceMgr, &Fixes);
      continue;
    }
    UsingDeclarations.push_back(UsingDeclaration(AnnotatedLines[I], Label));
  }
  endUsingDeclarationBlock(&UsingDeclarations, SourceMgr, &Fixes);
  return Fixes;
}

} // namespace format
} // namespace clang
