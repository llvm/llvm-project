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
#include "clang/AST/DeclCXX.h"
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
  const NamedDecl *SelectedAlias = nullptr;
  std::string AliasName;
  SourceRange AliasDeclToSkip;
  llvm::DenseSet<const NamedDecl *> RewriteTargets;
};
REGISTER_TWEAK(ReplaceQualifiedWithAlias)

bool collectUsingTargets(const UsingDecl *UD,
                         llvm::DenseSet<const NamedDecl *> &Targets) {
  if (!isNamespaceQualifier(UD->getQualifierLoc()))
    return false;
  for (const auto *Shadow : UD->shadows()) {
    if (const auto *Canonical = canonicalDecl(Shadow->getTargetDecl()))
      Targets.insert(Canonical);
  }
  return !Targets.empty();
}

bool collectAliasTargets(const NamedDecl *Alias, const Tweak::Selection &Inputs,
                         llvm::DenseSet<const NamedDecl *> &Targets) {
  if (const auto *UD = llvm::dyn_cast<UsingDecl>(Alias))
    return collectUsingTargets(UD, Targets);

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
  }

  return !Targets.empty();
}

/// Computes the set of primary file/namespace DeclContexts visible for
/// unqualified lookup at (\p UseCtx, \p UseLoc), including namespaces nominated
/// by active using-directives before \p UseLoc.
llvm::SmallPtrSet<const DeclContext *, 8>
collectVisibleContexts(const DeclContext *UseCtx, SourceLocation UseLoc,
                       ParsedAST &AST) {
  llvm::SmallPtrSet<const DeclContext *, 8> Visible;
  for (const DeclContext *Ctx = UseCtx; Ctx; Ctx = Ctx->getParent()) {
    if (Ctx->isFileContext())
      Visible.insert(Ctx->getPrimaryContext());
  }

  auto &SM = AST.getSourceManager();
  bool Changed = true;
  while (Changed) {
    Changed = false;
    auto ScanDecl = [&](auto &&Self, const Decl *D) -> void {
      SourceLocation DLoc = SM.getExpansionLoc(D->getBeginLoc());
      if (DLoc.isValid() && UseLoc.isValid() &&
          SM.isBeforeInTranslationUnit(UseLoc, DLoc))
        return;

      if (const auto *UDD = llvm::dyn_cast<UsingDirectiveDecl>(D)) {
        if (Visible.contains(UDD->getDeclContext()->getPrimaryContext())) {
          if (const auto *Nom = UDD->getNominatedNamespace()) {
            if (Visible.insert(Nom->getPrimaryContext()).second)
              Changed = true;
          }
        }
        return;
      }

      if (llvm::isa<NamespaceDecl, LinkageSpecDecl, ExportDecl>(D)) {
        for (const Decl *Child : llvm::cast<DeclContext>(D)->decls())
          Self(Self, Child);
      }
    };

    for (const Decl *D : AST.getLocalTopLevelDecls())
      ScanDecl(ScanDecl, D);
  }
  return Visible;
}

const UsingDecl *findMatchingUsingDecl(
    const Decl *D, const SourceManager &SM, SourceLocation SelectionLoc,
    const llvm::SmallPtrSetImpl<const DeclContext *> &VisibleContexts,
    llvm::StringRef SpelledName,
    const llvm::DenseSet<const NamedDecl *> &SelectedTargets,
    llvm::DenseSet<const NamedDecl *> &MatchedTargets) {
  SourceLocation DLoc = SM.getExpansionLoc(D->getBeginLoc());
  if (DLoc.isValid() && SelectionLoc.isValid() &&
      SM.isBeforeInTranslationUnit(SelectionLoc, DLoc))
    return nullptr;

  if (const auto *UD = llvm::dyn_cast<UsingDecl>(D)) {
    if (!UD->getIdentifier())
      return nullptr;
    if (!SpelledName.empty() && UD->getName() != SpelledName)
      return nullptr;
    if (!VisibleContexts.contains(UD->getDeclContext()->getPrimaryContext()))
      return nullptr;
    SourceLocation UDLoc = UD->getLocation();
    if (UDLoc.isInvalid() ||
        SM.getFileID(SM.getExpansionLoc(UDLoc)) != SM.getMainFileID())
      return nullptr;

    llvm::DenseSet<const NamedDecl *> AliasTargets;
    if (!collectUsingTargets(UD, AliasTargets))
      return nullptr;

    for (const auto *Target : AliasTargets) {
      if (SelectedTargets.contains(Target)) {
        MatchedTargets = std::move(AliasTargets);
        return UD;
      }
    }
    return nullptr;
  }

  // Only descend into file/namespace-level scopes (skipping functions,
  // classes, and enums).
  if (llvm::isa<NamespaceDecl, LinkageSpecDecl, ExportDecl>(D)) {
    for (const Decl *Child : llvm::cast<DeclContext>(D)->decls()) {
      if (const auto *UD = findMatchingUsingDecl(
              Child, SM, SelectionLoc, VisibleContexts, SpelledName,
              SelectedTargets, MatchedTargets))
        return UD;
    }
  }
  return nullptr;
}

std::string ReplaceQualifiedWithAlias::title() const {
  return std::string(
      llvm::formatv("Replace qualified references with {0}", AliasName));
}

bool ReplaceQualifiedWithAlias::prepare(const Selection &Inputs) {
  // Reset all cached state before inspecting the new selection.
  SelectedAlias = nullptr;
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
      SelectedAlias = Alias;
      RewriteTargets = std::move(Targets);
      AliasName = Alias->getNameAsString();
      AliasDeclToSkip = Alias->getSourceRange();
      return true;
    }
  }

  if (!Prepared.HasQualifiedRef)
    return false;

  // Search the main file's top-level and namespace declarations for a
  // using-declaration that is visible at the selection and covers the selected
  // reference.
  auto &SM = Inputs.AST->getSourceManager();
  SourceLocation SelectionLoc =
      SM.getExpansionLoc(Prepared.QualifierToRemove.getBeginLoc());
  const DeclContext *SelectionCtx =
      &Inputs.ASTSelection.commonAncestor()->getDeclContext();
  auto VisibleContexts =
      collectVisibleContexts(SelectionCtx, SelectionLoc, *Inputs.AST);

  llvm::DenseSet<const NamedDecl *> AliasTargets;
  for (const Decl *D : Inputs.AST->getLocalTopLevelDecls()) {
    if (const auto *UD = findMatchingUsingDecl(
            D, SM, SelectionLoc, VisibleContexts, Prepared.SpelledName,
            Prepared.Targets, AliasTargets)) {
      SelectedAlias = UD;
      RewriteTargets = std::move(AliasTargets);
      AliasName = UD->getNameAsString();
      AliasDeclToSkip = UD->getSourceRange();
      return true;
    }
  }

  return false;
}

Expected<Tweak::Effect>
ReplaceQualifiedWithAlias::apply(const Selection &Inputs) {
  auto &SM = Inputs.AST->getSourceManager();
  const auto &LangOpts = Inputs.AST->getLangOpts();

  // prepare() must have populated the alias name and target set.
  if (!SelectedAlias || AliasName.empty() || RewriteTargets.empty())
    return error("Incomplete prepared state for ReplaceQualifiedWithAlias");

  // Skip the selected alias declaration itself so we do not rewrite the alias
  // name inside its own definition.
  SourceLocation SkipBegin = AliasDeclToSkip.getBegin();
  SourceLocation SkipEnd = AliasDeclToSkip.getEnd();
  if (SkipBegin.isValid())
    SkipBegin = SM.getExpansionLoc(SkipBegin);
  if (SkipEnd.isValid())
    SkipEnd = SM.getExpansionLoc(SkipEnd);

  SourceLocation AliasLoc = SM.getExpansionLoc(SelectedAlias->getBeginLoc());
  const DeclContext *AliasPrimaryCtx =
      SelectedAlias->getDeclContext()->getPrimaryContext();
  const bool AliasAtTU = llvm::isa<TranslationUnitDecl>(AliasPrimaryCtx);

  tooling::Replacements Repls;
  auto RewriteRef = [&](const QualifiedReference &Ref) {
    if (SkipBegin.isValid() && SkipEnd.isValid() &&
        SM.isPointWithin(Ref.NameLoc, SkipBegin, SkipEnd))
      return;
    if (AliasLoc.isValid() &&
        SM.isBeforeInTranslationUnit(Ref.QualifierLoc, AliasLoc))
      return;

    unsigned BeginOffset = SM.getFileOffset(Ref.QualifierLoc);
    SourceLocation EndLoc = getReferenceEnd(Ref.NameLoc, SM, LangOpts);
    if (BeginOffset > SM.getFileOffset(EndLoc))
      return;

    if (auto Err = Repls.add(tooling::Replacement(
            SM, Ref.QualifierLoc, SM.getFileOffset(EndLoc) - BeginOffset,
            AliasName)))
      consumeError(std::move(Err));
  };

  if (AliasAtTU) {
    forEachQualifiedReferenceToRewrite(Inputs, RewriteTargets, RewriteRef);
    return Effect::mainFileEdit(SM, std::move(Repls));
  }

  // When the alias is declared inside a namespace, only rewrite references in
  // declarations where that namespace is visible (either directly enclosed by
  // the namespace or via an active using-directive).
  auto ProcessDecl = [&](auto &&Self, const Decl *D) -> void {
    if (llvm::isa<NamespaceDecl, LinkageSpecDecl, ExportDecl>(D)) {
      const auto *DC = llvm::cast<DeclContext>(D);
      if (!AliasPrimaryCtx->Encloses(DC)) {
        for (const Decl *Child : DC->decls())
          Self(Self, Child);
        return;
      }
    }

    SourceLocation DLoc = SM.getExpansionLoc(D->getBeginLoc());
    const DeclContext *UseCtx = llvm::isa<DeclContext>(D)
                                    ? llvm::cast<DeclContext>(D)
                                    : D->getDeclContext();
    auto Visible = collectVisibleContexts(UseCtx, DLoc, *Inputs.AST);
    if (!Visible.contains(AliasPrimaryCtx))
      return;

    findExplicitReferences(
        D,
        [&](ReferenceLoc Ref) {
          if (!Ref.Qualifier || Ref.Targets.empty() || Ref.IsDecl)
            return;
          if (!isNamespaceQualifier(Ref.Qualifier))
            return;
          if (!referenceMatchesTargets(Ref, RewriteTargets))
            return;

          QualifiedReference QRef;
          QRef.QualifierLoc = Ref.Qualifier.getBeginLoc();
          QRef.NameLoc = Ref.NameLoc;
          if (!normalizeToMainFile(QRef.QualifierLoc, SM) ||
              !normalizeToMainFile(QRef.NameLoc, SM))
            return;

          RewriteRef(QRef);
        },
        Inputs.AST->getHeuristicResolver());
  };

  for (const Decl *D : Inputs.AST->getLocalTopLevelDecls())
    ProcessDecl(ProcessDecl, D);

  return Effect::mainFileEdit(SM, std::move(Repls));
}

} // namespace
} // namespace clangd
} // namespace clang
