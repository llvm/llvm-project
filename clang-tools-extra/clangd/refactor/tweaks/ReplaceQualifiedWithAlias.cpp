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
#include "llvm/ADT/SmallVector.h"
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

void collectUsingDecls(const DeclContext *DC,
                       llvm::SmallVectorImpl<const UsingDecl *> &Out) {
  for (const Decl *D : DC->decls()) {
    if (const auto *UD = llvm::dyn_cast<UsingDecl>(D))
      Out.push_back(UD);
    if (const auto *Nested = llvm::dyn_cast<DeclContext>(D))
      collectUsingDecls(Nested, Out);
  }
}

// Returns the end of a qualified reference, extending past any template
// argument list so rewrites replace the full spelled name.
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
  // Reset all cached state before inspecting the new selection.
  AliasName.clear();
  AliasDeclToSkip = SourceRange();
  RewriteTargets.clear();

  auto *Node = Inputs.ASTSelection.commonAncestor();
  if (!Node)
    return false;

  auto SetFromAlias = [&](const NamedDecl *Alias) -> bool {
    if (!Alias->getIdentifier())
      return false;

    llvm::DenseSet<const NamedDecl *> Targets;

    ReferenceLoc QualifiedRef;
    bool FoundQualifiedRef = false;
    // Prefer the alias's own qualified reference when it exists, because it
    // gives us the exact namespace spelling that should be rewritten.
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

    if (FoundQualifiedRef) {
      for (const auto *Target : QualifiedRef.Targets) {
        if (const auto *Canonical = canonicalDecl(Target))
          Targets.insert(Canonical);
      }
    } else if (const auto *UD = llvm::dyn_cast<UsingDecl>(Alias)) {
      for (const auto *Shadow : UD->shadows()) {
        if (const auto *Canonical = canonicalDecl(Shadow->getTargetDecl()))
          Targets.insert(Canonical);
      }
    }

    if (Targets.empty())
      return false;

    // Cache the alias spelling and the canonical declarations it stands for.
    RewriteTargets = std::move(Targets);
    AliasName = Alias->getNameAsString();
    AliasDeclToSkip = Alias->getSourceRange();
    return true;
  };

  // First look for an alias declaration in the current selection or one of
  // its ancestors. This is the narrowest and most reliable source of truth.
  for (auto *N = Node; N; N = N->Parent) {
    if (const auto *TAD = N->ASTNode.get<TypeAliasDecl>(); TAD) {
      if (SetFromAlias(TAD))
        return true;
    } else if (const auto *UD = N->ASTNode.get<UsingDecl>(); UD) {
      if (SetFromAlias(UD))
        return true;
    }
  }

  llvm::DenseSet<const NamedDecl *> SelectedTargets;
  std::string SelectedName;
  bool FoundSelectedQualifiedRef = false;
  // If the selection is a qualified reference instead of an alias, derive the
  // target set from that reference and then search for a matching alias in the
  // file.
  for (const auto &D : Inputs.AST->getLocalTopLevelDecls()) {
    findExplicitReferences(
        D,
        [&](ReferenceLoc Ref) {
          if (FoundSelectedQualifiedRef || !Ref.Qualifier ||
              Ref.Targets.empty())
            return;
          if (!isNamespaceQualifier(Ref.Qualifier))
            return;

          auto &SM = Inputs.AST->getSourceManager();
          const auto &LangOpts = Inputs.AST->getLangOpts();

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

          unsigned BeginOffset = SM.getFileOffset(QualifierLoc);
          SourceLocation EndLoc = getReferenceEnd(NameLoc, SM, LangOpts);
          if (BeginOffset > SM.getFileOffset(EndLoc))
            return;

          if (Inputs.SelectionBegin < BeginOffset ||
              Inputs.SelectionBegin > SM.getFileOffset(EndLoc))
            return;

          // Record the referred-to name so we can prefer aliases with the same
          // identifier when several are visible.
          SelectedName =
              Lexer::getSourceText(CharSourceRange::getTokenRange(NameLoc), SM,
                                   LangOpts)
                  .str();
          for (const auto *Target : Ref.Targets) {
            if (const auto *Canonical = canonicalDecl(Target))
              SelectedTargets.insert(Canonical);
          }
          FoundSelectedQualifiedRef = true;
        },
        Inputs.AST->getHeuristicResolver());
    if (FoundSelectedQualifiedRef)
      break;
  }

  if (!FoundSelectedQualifiedRef)
    return false;

  llvm::SmallVector<const UsingDecl *> VisibleUsingDecls;
  collectUsingDecls(Inputs.AST->getASTContext().getTranslationUnitDecl(),
                    VisibleUsingDecls);
  auto &SM = Inputs.AST->getSourceManager();
  // Search visible using-declarations for one that shadows the same target
  // set, then reuse its alias spelling.
  for (const auto *UD : VisibleUsingDecls) {
    if (!UD->getIdentifier())
      continue;
    if (!SelectedName.empty() && UD->getName() != SelectedName)
      continue;
    SourceLocation UDLoc = UD->getLocation();
    if (UDLoc.isInvalid() ||
        SM.getFileID(SM.getExpansionLoc(UDLoc)) != SM.getMainFileID())
      continue;

    bool MatchesSelection = false;
    for (const auto *Shadow : UD->shadows()) {
      const auto *Canonical = canonicalDecl(Shadow->getTargetDecl());
      if (Canonical &&
          (SelectedTargets.empty() || SelectedTargets.contains(Canonical))) {
        MatchesSelection = true;
        break;
      }
    }
    if (!MatchesSelection)
      continue;

    if (SetFromAlias(UD))
      return true;
  }

  return false;
}

Expected<Tweak::Effect>
ReplaceQualifiedWithAlias::apply(const Selection &Inputs) {
  auto &SM = Inputs.AST->getSourceManager();
  const auto &LangOpts = Inputs.AST->getLangOpts();

  // prepare() must have populated the alias name and target set.
  if (AliasName.empty() || RewriteTargets.empty())
    return error("Incomplete prepared state for ReplaceQualifiedWithAlias");

  // Skip the selected alias declaration itself so we do not rewrite the alias
  // name inside its own definition.
  SourceLocation SkipBegin = AliasDeclToSkip.getBegin();
  SourceLocation SkipEnd = AliasDeclToSkip.getEnd();
  if (SkipBegin.isValid())
    SkipBegin = SM.getExpansionLoc(SkipBegin);
  if (SkipEnd.isValid())
    SkipEnd = SM.getExpansionLoc(SkipEnd);

  tooling::Replacements Repls;
  // Rewrite every explicit namespace-qualified reference that resolves to one
  // of the canonical declarations represented by the alias.
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

          // Leave the selected alias definition untouched.
          if (SkipBegin.isValid() && SkipEnd.isValid() &&
              SM.isPointWithin(NameLoc, SkipBegin, SkipEnd))
            return;

          unsigned BeginOffset = SM.getFileOffset(QualifierLoc);
          SourceLocation EndLoc = getReferenceEnd(NameLoc, SM, LangOpts);
          if (BeginOffset > SM.getFileOffset(EndLoc))
            return;

          // Replace the qualified spelling with the alias name only.
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
