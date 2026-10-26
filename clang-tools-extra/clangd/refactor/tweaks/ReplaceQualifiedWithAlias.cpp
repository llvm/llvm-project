//===--- ReplaceQualifiedWithAlias.cpp ---------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FindTarget.h"
#include "QualifiedNameRewrite.h"
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

void collectUsingDecls(const DeclContext *DC,
                       llvm::SmallVectorImpl<const UsingDecl *> &Out) {
  for (const Decl *D : DC->decls()) {
    if (const auto *UD = llvm::dyn_cast<UsingDecl>(D))
      Out.push_back(UD);
    if (const auto *Nested = llvm::dyn_cast<DeclContext>(D))
      collectUsingDecls(Nested, Out);
  }
}

bool collectAliasTargets(const NamedDecl *Alias, const Tweak::Selection &Inputs,
                         llvm::DenseSet<const NamedDecl *> &Targets) {
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
    return !Targets.empty();
  }

  if (const auto *UD = llvm::dyn_cast<UsingDecl>(Alias)) {
    for (const auto *Shadow : UD->shadows()) {
      if (const auto *Canonical = canonicalDecl(Shadow->getTargetDecl()))
        Targets.insert(Canonical);
    }
  }

  return !Targets.empty();
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

  const auto &Prepared = prepareQualifiedSelection(
      Inputs, QualifiedRewriteClient::ReplaceQualifiedWithAlias);

  // First look for an alias declaration in the current selection or one of
  // its ancestors. This is the narrowest and most reliable source of truth.
  if (const auto *Alias = Prepared.SelectedAliasDecl) {
    llvm::DenseSet<const NamedDecl *> Targets;
    if (collectAliasTargets(Alias, Inputs, Targets)) {
      RewriteTargets = std::move(Targets);
      AliasName = Alias->getNameAsString();
      AliasDeclToSkip = Alias->getSourceRange();
      return true;
    }
  }

  if (!Prepared.HasQualifiedRef)
    return false;

  // Search the main file's top-level declarations for a using-declaration that
  // covers the selected reference. Walking getLocalTopLevelDecls() instead of
  // getTranslationUnitDecl() avoids visiting declarations from included
  // headers.
  auto &SM = Inputs.AST->getSourceManager();
  llvm::SmallVector<const UsingDecl *> FileUsingDecls;
  for (const Decl *D : Inputs.AST->getLocalTopLevelDecls()) {
    if (const auto *UD = llvm::dyn_cast<UsingDecl>(D))
      FileUsingDecls.push_back(UD);
    if (const auto *Nested = llvm::dyn_cast<DeclContext>(D))
      collectUsingDecls(Nested, FileUsingDecls);
  }
  for (const auto *UD : FileUsingDecls) {
    if (!UD->getIdentifier())
      continue;
    if (!Prepared.SpelledName.empty() && UD->getName() != Prepared.SpelledName)
      continue;
    SourceLocation UDLoc = UD->getLocation();
    if (UDLoc.isInvalid() ||
        SM.getFileID(SM.getExpansionLoc(UDLoc)) != SM.getMainFileID())
      continue;

    llvm::DenseSet<const NamedDecl *> AliasTargets;
    if (!collectAliasTargets(UD, Inputs, AliasTargets))
      continue;

    bool MatchesSelection = false;
    for (const auto *Target : AliasTargets) {
      if (Prepared.Targets.contains(Target)) {
        MatchesSelection = true;
        break;
      }
    }
    if (!MatchesSelection)
      continue;

    RewriteTargets = std::move(AliasTargets);
    AliasName = UD->getNameAsString();
    AliasDeclToSkip = UD->getSourceRange();
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
  forEachQualifiedReferenceToRewrite(
      Inputs, RewriteTargets, [&](const QualifiedReference &Ref) {
        // Leave the selected alias definition untouched.
        if (SkipBegin.isValid() && SkipEnd.isValid() &&
            SM.isPointWithin(Ref.NameLoc, SkipBegin, SkipEnd))
          return;

        unsigned BeginOffset = SM.getFileOffset(Ref.QualifierLoc);
        SourceLocation EndLoc = getReferenceEnd(Ref.NameLoc, SM, LangOpts);
        if (BeginOffset > SM.getFileOffset(EndLoc))
          return;

        // Replace the qualified spelling with the alias name only.
        if (auto Err = Repls.add(tooling::Replacement(
                SM, Ref.QualifierLoc, SM.getFileOffset(EndLoc) - BeginOffset,
                AliasName)))
          consumeError(std::move(Err));
      });

  return Effect::mainFileEdit(SM, std::move(Repls));
}

} // namespace
} // namespace clangd
} // namespace clang
