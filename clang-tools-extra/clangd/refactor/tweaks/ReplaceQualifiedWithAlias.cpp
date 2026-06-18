//===--- ReplaceQualifiedWithAlias.cpp ---------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FindTarget.h"
#include "refactor/Tweak.h"
#include "support/Logger.h"

#include "clang/AST/Decl.h"
#include "clang/AST/NestedNameSpecifier.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include <string>

namespace clang {
namespace clangd {
namespace {

class ReplaceQualifiedWithAlias : public Tweak {
public:
  const char *id() const override;

  bool prepare(const Selection &Inputs) override;
  Expected<Effect> apply(const Selection &Inputs) override;
  std::string title() const override;
  llvm::StringLiteral kind() const override {
    return CodeAction::REFACTOR_KIND;
  }

private:
  std::string AliasName;
  SourceRange AliasDeclToSkip;
  llvm::DenseSet<const NamedDecl *> RewriteTargets;
};
REGISTER_TWEAK(ReplaceQualifiedWithAlias)

const NamedDecl *canonicalDecl(const NamedDecl *D) {
  return D ? llvm::dyn_cast<NamedDecl>(D->getCanonicalDecl()) : nullptr;
}

bool isNamespaceQualifier(NestedNameSpecifierLoc Qualifier) {
  return Qualifier && Qualifier.getNestedNameSpecifier().getKind() ==
                          NestedNameSpecifier::Kind::Namespace;
}

SourceLocation getReferenceEnd(SourceLocation NameLoc, const SourceManager &SM,
                               const LangOptions &LangOpts) {
  std::optional<Token> Tok = Lexer::findNextToken(NameLoc, SM, LangOpts);
  if (!Tok || Tok->isNot(tok::less))
    return Lexer::getLocForEndOfToken(NameLoc, 0, SM, LangOpts);

  unsigned Depth = 0;
  while (Tok) {
    switch (Tok->getKind()) {
    case tok::less:
      ++Depth;
      break;
    case tok::greater:
      if (Depth == 0)
        return Lexer::getLocForEndOfToken(Tok->getLocation(), 0, SM, LangOpts);
      --Depth;
      if (Depth == 0)
        return Lexer::getLocForEndOfToken(Tok->getLocation(), 0, SM, LangOpts);
      break;
    case tok::greatergreater:
      if (Depth <= 1)
        return Lexer::getLocForEndOfToken(Tok->getLocation(), 0, SM, LangOpts);
      Depth -= 2;
      if (Depth == 0)
        return Lexer::getLocForEndOfToken(Tok->getLocation(), 0, SM, LangOpts);
      break;
    default:
      if (Depth == 0)
        return Lexer::getLocForEndOfToken(Tok->getLocation(), 0, SM, LangOpts);
      break;
    }
    Tok = Lexer::findNextToken(Tok->getLocation(), SM, LangOpts);
  }
  return Lexer::getLocForEndOfToken(NameLoc, 0, SM, LangOpts);
}

std::string ReplaceQualifiedWithAlias::title() const {
  return std::string(
      llvm::formatv("Replace qualified references with {0}", AliasName));
}

bool ReplaceQualifiedWithAlias::prepare(const Selection &Inputs) {
  AliasName.clear();
  AliasDeclToSkip = SourceRange();
  RewriteTargets.clear();

  auto *Node = Inputs.ASTSelection.commonAncestor();
  if (!Node)
    return false;

  for (auto *N = Node; N; N = N->Parent) {
    const auto *Alias = N->ASTNode.get<TypeAliasDecl>();
    if (!Alias)
      continue;
    if (!Alias->getIdentifier())
      return false;

    ReferenceLoc QualifiedRef;
    bool FoundQualifiedRef = false;
    findExplicitReferences(
        Alias,
        [&](ReferenceLoc Ref) {
          if (FoundQualifiedRef || !Ref.Qualifier || Ref.Targets.empty())
            return;
          if (!isNamespaceQualifier(Ref.Qualifier))
            return;
          QualifiedRef = std::move(Ref);
          FoundQualifiedRef = true;
        },
        Inputs.AST->getHeuristicResolver());

    if (!FoundQualifiedRef)
      return false;

    for (const auto *Target : QualifiedRef.Targets) {
      if (const auto *Canonical = canonicalDecl(Target))
        RewriteTargets.insert(Canonical);
    }
    if (RewriteTargets.empty())
      return false;

    AliasName = Alias->getNameAsString();
    AliasDeclToSkip = Alias->getSourceRange();
    return true;
  }

  return false;
}

Expected<Tweak::Effect>
ReplaceQualifiedWithAlias::apply(const Selection &Inputs) {
  auto &SM = Inputs.AST->getSourceManager();
  const auto &LangOpts = Inputs.AST->getLangOpts();

  if (AliasName.empty() || RewriteTargets.empty())
    return error("Incomplete prepared state for ReplaceQualifiedWithAlias");

  SourceLocation SkipBegin = AliasDeclToSkip.getBegin();
  SourceLocation SkipEnd = AliasDeclToSkip.getEnd();
  if (SkipBegin.isValid())
    SkipBegin = SM.getExpansionLoc(SkipBegin);
  if (SkipEnd.isValid())
    SkipEnd = SM.getExpansionLoc(SkipEnd);

  tooling::Replacements Repls;
  for (const auto &D : Inputs.AST->getLocalTopLevelDecls()) {
    findExplicitReferences(
        D,
        [&](ReferenceLoc Ref) {
          if (!Ref.Qualifier || Ref.Targets.empty() || Ref.IsDecl)
            return;
          if (!isNamespaceQualifier(Ref.Qualifier))
            return;

          bool MatchesTarget = false;
          for (const auto *Target : Ref.Targets) {
            if (const auto *Canonical = canonicalDecl(Target);
                Canonical && RewriteTargets.contains(Canonical)) {
              MatchesTarget = true;
              break;
            }
          }
          if (!MatchesTarget)
            return;

          SourceLocation QualifierLoc = Ref.Qualifier.getBeginLoc();
          SourceLocation NameLoc = Ref.NameLoc;
          if (QualifierLoc.isMacroID()) {
            if (!SM.isMacroArgExpansion(QualifierLoc))
              return;
            QualifierLoc = SM.getFileLoc(QualifierLoc);
          }
          if (NameLoc.isMacroID()) {
            if (!SM.isMacroArgExpansion(NameLoc))
              return;
            NameLoc = SM.getFileLoc(NameLoc);
          }
          if (SM.getFileID(QualifierLoc) != SM.getMainFileID() ||
              SM.getFileID(NameLoc) != SM.getMainFileID())
            return;

          if (SkipBegin.isValid() && SkipEnd.isValid() &&
              SM.isPointWithin(NameLoc, SkipBegin, SkipEnd))
            return;

          unsigned BeginOffset = SM.getFileOffset(QualifierLoc);
          SourceLocation EndLoc = getReferenceEnd(NameLoc, SM, LangOpts);
          if (BeginOffset > SM.getFileOffset(EndLoc))
            return;

          if (auto Err = Repls.add(tooling::Replacement(
                  SM, QualifierLoc, SM.getFileOffset(EndLoc) - BeginOffset,
                  AliasName)))
            consumeError(std::move(Err));
        },
        Inputs.AST->getHeuristicResolver());
  }

  return Effect::mainFileEdit(SM, std::move(Repls));
}

} // namespace
} // namespace clangd
} // namespace clang
