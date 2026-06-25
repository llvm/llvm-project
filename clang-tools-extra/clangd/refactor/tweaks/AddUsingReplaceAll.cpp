//===--- AddUsingReplaceAll.cpp ---------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AST.h"
#include "Config.h"
#include "FindTarget.h"
#include "SourceCode.h"
#include "refactor/Tweak.h"
#include "support/Logger.h"

#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/Expr.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Tooling/Core/Replacement.h"
#include "clang/Tooling/Syntax/Tokens.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <tuple>
#include <utility>

namespace clang {
namespace clangd {

/// Tweak for removing the full namespace qualifier under the cursor on
/// DeclRefExpr and types and adding "using" statement instead. This tweak
/// replaces all occurrences of the qualified symbol in the file, not just the
/// one under the cursor, but it also requires more preparation and is more
/// expensive to compute, so it is hidden behind a separate action from
/// AddUsing, which only removes the qualifier under the cursor.
class AddUsingReplaceAll : public Tweak {
public:
  const char *id() const override;

  bool prepare(const Selection &Inputs) override;
  Expected<Effect> apply(const Selection &Inputs) override;
  std::string title() const override;
  llvm::StringLiteral kind() const override {
    return CodeAction::REFACTOR_KIND;
  }

private:
  NestedNameSpecifierLoc QualifierToRemove;
  std::string QualifierToSpell;
  llvm::StringRef SpelledQualifier;
  llvm::StringRef SpelledName;
  // The selected declaration must stay after the inserted using-declaration.
  SourceLocation MustInsertAfterLoc;
  // Canonical declarations whose explicit qualified references will be
  // rewritten in the main file.
  llvm::DenseSet<const NamedDecl *> RewriteTargets;
};
REGISTER_TWEAK(AddUsingReplaceAll)

std::string AddUsingReplaceAll::title() const {
  return std::string(llvm::formatv("Add using-declaration for {0} and replace "
                                   "all qualified references in this file",
                                   SpelledName));
}

/// Collects all UsingDecls visible at the selection by walking the enclosing
/// DeclContext chain directly, sorted by source location. This avoids a full
/// AST traversal and only visits O(depth * declarations-per-scope) nodes.
///
/// A using-declaration declared in a scope S is visible at SelectionCtx iff
/// S is an ancestor of SelectionCtx, i.e., exactly the contexts we walk when
/// ascending via getLexicalParent().
static std::vector<const UsingDecl *>
collectEnclosingUsings(const DeclContext *SelectionCtx,
                       const SourceManager &SM) {
  std::vector<const UsingDecl *> Result;
  for (const DeclContext *Ctx = SelectionCtx; Ctx;
       Ctx = Ctx->getLexicalParent()) {
    for (const Decl *D : Ctx->decls()) {
      const auto *UD = dyn_cast<UsingDecl>(D);
      if (!UD)
        continue;
      if (SM.getFileID(UD->getUsingLoc()) != SM.getMainFileID())
        continue;
      Result.push_back(UD);
    }
  }
  // Merge into a single sorted list so findInsertionPoint can break early.
  llvm::sort(Result, [&](const UsingDecl *A, const UsingDecl *B) {
    return SM.isBeforeInTranslationUnit(A->getUsingLoc(), B->getUsingLoc());
  });
  return Result;
}

struct InsertionPointData {
  SourceLocation Loc;
  std::string Suffix;
  bool AlwaysFullyQualify = false;
};

static const NamespaceDecl *
getNamespaceFromQualifier(NestedNameSpecifier Qualifier) {
  if (!Qualifier || Qualifier.getKind() != NestedNameSpecifier::Kind::Namespace)
    return nullptr;
  return dyn_cast_or_null<NamespaceDecl>(
      Qualifier.getAsNamespaceAndPrefix().Namespace);
}

static llvm::Expected<InsertionPointData>
findInsertionPoint(const Tweak::Selection &Inputs,
                   const NestedNameSpecifierLoc &QualifierToRemove,
                   const llvm::StringRef Name,
                   const SourceLocation MustInsertAfterLoc) {
  auto &SM = Inputs.AST->getSourceManager();

  const auto *TargetNamespace =
      getNamespaceFromQualifier(QualifierToRemove.getNestedNameSpecifier());
  if (!TargetNamespace)
    return error("Qualifier is not a valid namespace qualifier");

  SourceLocation LastUsingLoc;
  const std::vector<const UsingDecl *> Usings = collectEnclosingUsings(
      &Inputs.ASTSelection.commonAncestor()->getDeclContext(), SM);

  auto IsValidPoint = [&](const SourceLocation Loc) {
    return MustInsertAfterLoc.isInvalid() ||
           SM.isBeforeInTranslationUnit(MustInsertAfterLoc, Loc);
  };

  bool AlwaysFullyQualify = true;
  for (const auto *U : Usings) {
    if (!U->getQualifier().isFullyQualified())
      AlwaysFullyQualify = false;

    if (const auto *Namespace = getNamespaceFromQualifier(U->getQualifier())) {
      if (Namespace->getCanonicalDecl() ==
              TargetNamespace->getCanonicalDecl() &&
          U->getName() == Name)
        return InsertionPointData();
    }

    LastUsingLoc = U->getUsingLoc();
  }
  if (LastUsingLoc.isValid() && IsValidPoint(LastUsingLoc)) {
    InsertionPointData Out;
    Out.Loc = LastUsingLoc;
    Out.AlwaysFullyQualify = AlwaysFullyQualify;
    return Out;
  }

  const DeclContext *ParentDeclCtx =
      &Inputs.ASTSelection.commonAncestor()->getDeclContext();
  while (ParentDeclCtx && !ParentDeclCtx->isFileContext())
    ParentDeclCtx = ParentDeclCtx->getLexicalParent();

  if (auto *ND = llvm::dyn_cast_or_null<NamespaceDecl>(ParentDeclCtx)) {
    auto Toks = Inputs.AST->getTokens().expandedTokens(ND->getSourceRange());
    const auto *Tok = llvm::find_if(Toks, [](const syntax::Token &Tok) {
      return Tok.kind() == tok::l_brace;
    });
    if (Tok == Toks.end() || Tok->endLocation().isInvalid())
      return error("Namespace with no {{");
    if (!Tok->endLocation().isMacroID() && IsValidPoint(Tok->endLocation())) {
      InsertionPointData Out;
      Out.Loc = Tok->endLocation();
      Out.Suffix = "\n";
      return Out;
    }
  }

  auto TLDs = Inputs.AST->getLocalTopLevelDecls();
  for (const auto &TLD : TLDs) {
    if (!IsValidPoint(TLD->getBeginLoc()))
      continue;
    InsertionPointData Out;
    Out.Loc = SM.getExpansionLoc(TLD->getBeginLoc());
    Out.Suffix = "\n\n";
    return Out;
  }
  return error("Cannot find place to insert \"using\"");
}

static bool isNamespaceForbidden(const Tweak::Selection &Inputs,
                                 NestedNameSpecifier Namespace) {
  const auto *NS =
      dyn_cast<NamespaceDecl>(Namespace.getAsNamespaceAndPrefix().Namespace);
  if (!NS)
    return true;
  std::string NamespaceStr = printNamespaceScope(*NS);

  for (StringRef Banned : Config::current().Style.FullyQualifiedNamespaces) {
    StringRef PrefixMatch = NamespaceStr;
    if (PrefixMatch.consume_front(Banned) && PrefixMatch.consume_front("::"))
      return true;
  }

  return false;
}

static std::string getNNSLAsString(NestedNameSpecifierLoc NNSL,
                                   const PrintingPolicy &Policy) {
  std::string Out;
  llvm::raw_string_ostream OutStream(Out);
  NNSL.getNestedNameSpecifier().print(OutStream, Policy);
  return OutStream.str();
}

static const NamedDecl *canonicalDecl(const NamedDecl *D) {
  return D ? llvm::dyn_cast<NamedDecl>(D->getCanonicalDecl()) : nullptr;
}

bool AddUsingReplaceAll::prepare(const Selection &Inputs) {
  auto &SM = Inputs.AST->getSourceManager();
  const auto &TB = Inputs.AST->getTokens();

  // Clear any state from a previous invocation before analyzing the new
  // selection.
  QualifierToRemove = NestedNameSpecifierLoc();
  QualifierToSpell.clear();
  SpelledQualifier = llvm::StringRef();
  SpelledName = llvm::StringRef();
  MustInsertAfterLoc = SourceLocation();
  RewriteTargets.clear();

  if (isHeaderFile(SM.getFileEntryRefForID(SM.getMainFileID())->getName(),
                   Inputs.AST->getLangOpts()))
    return false;

  // Normalize the selection to the outermost AST node that still represents
  // the qualified spelling we want to rewrite.
  auto *Node = Inputs.ASTSelection.commonAncestor();
  if (!Node)
    return false;

  for (; Node->Parent; Node = Node->Parent) {
    if (Node->ASTNode.get<NestedNameSpecifierLoc>())
      continue;
    if (auto *T = Node->ASTNode.get<TypeLoc>()) {
      if (Node->Parent->ASTNode.get<NestedNameSpecifierLoc>())
        continue;
      if (isa<TagType, TemplateSpecializationType, TypedefType, UsingType,
              UnresolvedUsingType>(T->getTypePtr()))
        break;
      if (Node->Parent->ASTNode.get<TypeLoc>())
        continue;
    }
    break;
  }
  if (!Node)
    return false;

  SourceRange SpelledNameRange;
  // Capture the qualified spelling, the namespace qualifier to remove, and
  // the declaration that constrains where a new using-declaration may go.
  if (auto *D = Node->ASTNode.get<DeclRefExpr>()) {
    if (D->getDecl()->getIdentifier()) {
      QualifierToRemove = D->getQualifierLoc();
      SpelledNameRange = D->getSourceRange();
      if (auto AngleLoc = D->getLAngleLoc(); AngleLoc.isValid())
        SpelledNameRange.setEnd(AngleLoc.getLocWithOffset(-1));
      MustInsertAfterLoc = D->getDecl()->getBeginLoc();
      if (const auto *Canonical = canonicalDecl(D->getDecl()))
        RewriteTargets.insert(Canonical);
    }
  } else if (auto *T = Node->ASTNode.get<TypeLoc>()) {
    switch (T->getTypeLocClass()) {
    case TypeLoc::TemplateSpecialization: {
      auto TL = T->castAs<TemplateSpecializationTypeLoc>();
      QualifierToRemove = TL.getQualifierLoc();
      if (!QualifierToRemove)
        break;
      SpelledNameRange = TL.getTemplateNameLoc();
      if (auto *TD = TL.getTypePtr()->getTemplateName().getAsTemplateDecl(
              /*IgnoreDeduced=*/true)) {
        MustInsertAfterLoc = TD->getBeginLoc();
        if (const auto *Canonical = canonicalDecl(TD))
          RewriteTargets.insert(Canonical);
      }
      break;
    }
    case TypeLoc::Enum:
    case TypeLoc::Record:
    case TypeLoc::InjectedClassName: {
      auto TL = T->castAs<TagTypeLoc>();
      QualifierToRemove = TL.getQualifierLoc();
      if (!QualifierToRemove)
        break;
      SpelledNameRange = TL.getNameLoc();
      MustInsertAfterLoc = TL.getDecl()->getBeginLoc();
      if (const auto *Canonical = canonicalDecl(TL.getDecl()))
        RewriteTargets.insert(Canonical);
      break;
    }
    case TypeLoc::Typedef: {
      auto TL = T->castAs<TypedefTypeLoc>();
      QualifierToRemove = TL.getQualifierLoc();
      if (!QualifierToRemove)
        break;
      SpelledNameRange = TL.getNameLoc();
      MustInsertAfterLoc = TL.getDecl()->getBeginLoc();
      if (const auto *Canonical = canonicalDecl(TL.getDecl()))
        RewriteTargets.insert(Canonical);
      break;
    }
    case TypeLoc::UnresolvedUsing: {
      auto TL = T->castAs<UnresolvedUsingTypeLoc>();
      QualifierToRemove = TL.getQualifierLoc();
      if (!QualifierToRemove)
        break;
      SpelledNameRange = TL.getNameLoc();
      MustInsertAfterLoc = TL.getDecl()->getBeginLoc();
      if (const auto *Canonical = canonicalDecl(TL.getDecl()))
        RewriteTargets.insert(Canonical);
      break;
    }
    case TypeLoc::Using: {
      auto TL = T->castAs<UsingTypeLoc>();
      QualifierToRemove = TL.getQualifierLoc();
      if (!QualifierToRemove)
        break;
      SpelledNameRange = TL.getNameLoc();
      MustInsertAfterLoc = TL.getDecl()->getBeginLoc();
      if (const auto *Canonical = canonicalDecl(TL.getDecl()))
        RewriteTargets.insert(Canonical);
      break;
    }
    default:
      break;
    }
    // For type spellings, the qualifier is part of the type name range rather
    // than a separate expression node.
    if (QualifierToRemove)
      SpelledNameRange.setBegin(QualifierToRemove.getBeginLoc());
  }

  // Only handle fully namespace-qualified spellings that can be rewritten
  // consistently across the file.
  if (!QualifierToRemove || RewriteTargets.empty() ||
      QualifierToRemove.getNestedNameSpecifier().getKind() !=
          NestedNameSpecifier::Kind::Namespace ||
      isNamespaceForbidden(Inputs, QualifierToRemove.getNestedNameSpecifier()))
    return false;

  // Macro body expansions and cross-file spellings are too ambiguous for this
  // refactoring.
  if (SM.isMacroBodyExpansion(QualifierToRemove.getBeginLoc()) ||
      !SM.isWrittenInSameFile(QualifierToRemove.getBeginLoc(),
                              QualifierToRemove.getEndLoc()))
    return false;

  auto SpelledTokens =
      TB.spelledForExpanded(TB.expandedTokens(SpelledNameRange));
  if (!SpelledTokens)
    return false;
  auto SpelledRange =
      syntax::Token::range(SM, SpelledTokens->front(), SpelledTokens->back());
  std::tie(SpelledQualifier, SpelledName) =
      splitQualifiedName(SpelledRange.text(SM));
  // Keep the exact source spelling for the qualifier so the inserted using
  // matches the style of the original code.
  QualifierToSpell = getNNSLAsString(
      QualifierToRemove, Inputs.AST->getASTContext().getPrintingPolicy());
  if (!llvm::StringRef(QualifierToSpell).ends_with(SpelledQualifier) ||
      SpelledName.empty())
    return false;
  return true;
}

Expected<Tweak::Effect> AddUsingReplaceAll::apply(const Selection &Inputs) {
  auto &SM = Inputs.AST->getSourceManager();

  // Fail loudly if apply() is reached without the state produced by prepare().
  const auto *TargetNamespace =
      getNamespaceFromQualifier(QualifierToRemove.getNestedNameSpecifier());
  if (!TargetNamespace)
    return error("Invalid namespace qualifier in prepared state");
  if (SpelledName.empty() || QualifierToSpell.empty() || RewriteTargets.empty())
    return error("Incomplete prepared state for AddUsingReplaceAll");

  tooling::Replacements Repls;

  // Find a legal insertion point based on nearby using-declarations, namespace
  // scope, or the start of the file.
  auto InsertionPoint = findInsertionPoint(Inputs, QualifierToRemove,
                                           SpelledName, MustInsertAfterLoc);
  if (!InsertionPoint) {
    std::string Msg = llvm::toString(InsertionPoint.takeError());
    elog("AddUsingReplaceAll: {0}", Msg);
    return error(Msg);
  }

  if (InsertionPoint->Loc.isValid()) {
    // Emit the new using-declaration with any extra qualification required by
    // existing local style.
    std::string UsingText;
    llvm::raw_string_ostream UsingTextStream(UsingText);
    UsingTextStream << "using ";
    if (InsertionPoint->AlwaysFullyQualify &&
        !QualifierToRemove.getNestedNameSpecifier().isFullyQualified())
      UsingTextStream << "::";
    UsingTextStream << QualifierToSpell << SpelledName << ";"
                    << InsertionPoint->Suffix;

    assert(SM.getFileID(InsertionPoint->Loc) == SM.getMainFileID());
    if (auto Err = Repls.add(tooling::Replacement(SM, InsertionPoint->Loc, 0,
                                                  UsingTextStream.str())))
      return std::move(Err);
  }

  bool HasReplacementError = false;
  std::string ReplacementError;
  // Walk every top-level declaration in the file and rewrite only the
  // explicit references that resolve to one of the prepared targets.
  for (const auto &D : Inputs.AST->getLocalTopLevelDecls()) {
    findExplicitReferences(
        D,
        [&](ReferenceLoc Ref) {
          if (HasReplacementError)
            return;
          if (!Ref.Qualifier || Ref.Targets.empty() || Ref.IsDecl)
            return;
          if (!getNamespaceFromQualifier(
                  Ref.Qualifier.getNestedNameSpecifier()))
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

          unsigned BeginOffset = SM.getFileOffset(QualifierLoc);
          unsigned EndOffset = SM.getFileOffset(NameLoc);
          if (BeginOffset >= EndOffset)
            return;

          // Drop the namespace qualifier, leaving the referenced name in
          // place.
          if (auto Err = Repls.add(tooling::Replacement(
                  SM, QualifierLoc, EndOffset - BeginOffset, ""))) {
            ReplacementError = llvm::toString(std::move(Err));
            elog("AddUsingReplaceAll: {0}", ReplacementError);
            HasReplacementError = true;
            return;
          }
        },
        Inputs.AST->getHeuristicResolver());
  }

  if (HasReplacementError)
    return error(ReplacementError);

  return Effect::mainFileEdit(SM, std::move(Repls));
}

} // namespace clangd
} // namespace clang
